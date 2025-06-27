import numpy as np
import pandas as pd
from copy import deepcopy
import time

import torch
import torch.optim as optim
from torch import nn

import hypnettorch.utils.hnet_regularizer as hreg

from method.method_abc import MethodABC
from method.utils import mixup_data
from method.loss_functions import mixup_criterion, calculate_worst_case_loss

from typing import Tuple

class SHIELD(MethodABC):
    """
    SHIELD is a continual learning method that incorporates interval mixup technique, 
    and hypernetwork (hnet) regularization to mitigate catastrophic forgetting 
    across sequential tasks. The method leverages the hnet to generate task-specific 
    weights and applies a combination of fit and specification losses, balanced by a scheduled 
    kappa parameter, to optimize performance on both current and previous tasks.
    Args:
        beta (float): Regularization strength for previous tasks.
        mixup_alpha (float): Alpha parameter for the mixup data augmentation.
        no_iterations (int): Total number of training iterations per task.
    Attributes:
        beta (float): Regularization strength for previous tasks.
        mixup_alpha (float): Alpha parameter for the mixup data augmentation.
        no_iterations (int): Total number of training iterations per task.
        base_epsilon (float): Base perturbation value, typically set from the module.
        current_epsilon (float): Current perturbation value, scheduled during training.
        current_kappa (float): Current kappa value, scheduled during training.
        regularization_targets (Any): Targets for regularization, set for tasks > 0.
        criterion (nn.Module): Loss function used for classification.
    Methods:
        schedule_epsilon(iteration: int) -> None:
        schedule_kappa(iteration: int) -> float:
        setup_task(task_id: int) -> None:
            Prepares the model for a new task, setting regularization targets and resetting schedules.
        forward(x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs a forward pass with mixup augmentation, scheduled perturbation, and regularization.
    """
   

    def __init__(self, 
                 beta: float,
                 mixup_alpha: float,
                 no_iterations: int
                 ):
        super().__init__()

        self.beta = beta
        self.mixup_alpha = mixup_alpha
        self.no_iterations = no_iterations

        self.base_epsilon = self.module.epsilon
        self.current_epsilon = 0.0
        self.current_kappa = 1.0

        self.regularization_targets = None
        self.criterion = nn.CrossEntropyLoss()

    def schedule_epsilon(self, iteration: int) -> None:
        """
        Schedules the perturbation epsilon based on the current iteration.
        The epsilon increases linearly until it reaches the base epsilon.
        """
        if iteration <= self.no_iterations // 2:
            self.current_epsilon = 2 * iteration / self.no_iterations * self.base_epsilon
        else:
            self.current_epsilon = self.base_epsilon

    def schedule_kappa(self, iteration: int) -> float:
        """
        Schedules the kappa value based on the current iteration.
        The kappa decreases linearly until it reaches 0.5.
        """
        if iteration <= self.no_iterations // 2:
            self.current_kappa = 1 - (iteration / self.no_iterations)
        else:
            self.current_kappa = 0.5

    def setup_task(self, task_id: int) -> None:
        """
        Prepares the model for training or evaluation on a specific task.
        Args:
            task_id (int): The identifier of the task to set up.
        Side Effects:
            - If `task_id` is greater than 0:
                - Freezes the parameters of the previous task in the hypernetwork by setting
                  `requires_grad` to False for the corresponding conditional parameters.
                - Updates `self.regularization_targets` using the current targets from the
                  hypernetwork regularizer.
            - Resets `self.current_epsilon` to 0.0.
            - Sets `self.current_kappa` to 1.0.
        """

        if task_id > 0:
            self.hnet.conditional_params[task_id-1].requires_grad_(False)

            self.regularization_targets = hreg.get_current_targets(
                task_id, self.module.hnet
            )

        self.current_epsilon = 0.0
        self.current_kappa = 1.0
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the SHIELD method with mixup data augmentation and regularization.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            y (torch.Tensor): Ground truth labels tensor of shape (batch_size,).
            task_id (int): Identifier for the current task, used for conditional hypernetwork weights and regularization.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The computed loss for the current batch.
        Workflow:
            - Schedules epsilon and kappa if in training mode.
            - Obtains task-specific weights from the hypernetwork.
            - Applies mixup augmentation to the input and labels.
            - Computes predictions and their associated epsilon-transformed predictions.
            - Calculates lower and upper prediction bounds and selects the appropriate bound based on the mixup labels.
            - Computes two losses: specification loss (loss_spec) and fit loss (loss_fit) using the mixup criterion.
            - Calculates the worst-case loss using the lower and upper bounds.
            - Combines the losses using the current value of kappa.
            - Adds regularization loss if not the first task, using the hypernetwork regularizer.
            - Returns the total loss as the sum of the current task loss and the scaled regularization loss.
        """

        if self.module.hnet.training:
            self.schedule_epsilon()
            self.schedule_kappa()

        hnet_weights = self.hnet.forward(cond_id=task_id)

        # Apply mixup augmentation
        mixup_tensor_input, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
        eps_transformed = abs(2*lam-1.0) * self.current_epsilon

        # Calculate predictions
        prediction, eps_prediction = self.module.target_network.forward(
            mixup_tensor_input,
            epsilon=eps_transformed, 
            weights=hnet_weights
        )

        z_lower = prediction - eps_prediction
        z_upper = prediction + eps_prediction
        z = torch.where((nn.functional.one_hot(y_a, prediction.size(-1))).bool(), z_lower, z_upper)

        loss_spec = mixup_criterion(self.criterion, z, y_a, y_b, lam)
        loss_fit = mixup_criterion(self.criterion, prediction, y_a, y_b, lam)

        z = calculate_worst_case_loss(
            z_lower=z_lower,
            z_upper=z_upper,
            prediction=prediction,
            gt_output=y
        )

        loss_current_task = self.current_kappa * loss_fit + (1 -self.current_kappa) * loss_spec

        loss_regularization = 0.0
        if task_id > 0:
            loss_regularization = hreg.calc_fix_target_reg(
                self.module.hnet,
                task_id,
                targets=self.regularization_targets,
                mnet=self.module.target_network,
                prev_theta=None,
                prev_task_embs=None,
                inds_of_out_heads=None,
                batch_size=-1,
            )

        loss = loss_current_task + self.beta * loss_regularization / max(1, task_id)
        
        return loss