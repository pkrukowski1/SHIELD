import torch
from torch import nn
import hypnettorch.utils.hnet_regularizer as hreg

from method.method_abc import MethodABC
from method.utils import mixup_data
from model.model_abc import CLModuleABC
from method.interval_mixup_decay_rate import MixupEpsilonDecayRate

from typing import Tuple
from copy import deepcopy

class SHIELD(MethodABC):
    """
    SHIELD is a continual learning method that incorporates interval mixup technique, 
    and hypernetwork (hnet) regularization to mitigate catastrophic forgetting 
    across sequential tasks. The method leverages the hnet to generate task-specific 
    weights and applies a combination of fit and specification losses, balanced by a scheduled 
    kappa parameter, to optimize performance on both current and previous tasks.
    Args:
        module (CLModuleABC): Continual learning module object.
        epsilon (float): Perturbation radii.
        lr (float): Learning rate.
        use_lr_scheduler (bool): Flag to indicate if a learning rate scheduler should
            be used or not.
        beta (float): Regularization strength for previous tasks.
        mixup_alpha (float): Alpha parameter for the mixup data augmentation.
        no_iterations (int): Total number of training iterations per task.
        mixup_epsilon_decay (str): Decaying rate of an epsilon in Interval MixUp.
        final_kappa (float): Final value of weight for worst-case vs standard loss.
    Attributes:
        beta (float): Regularization strength for previous tasks.
        mixup_alpha (float): Alpha parameter for the mixup data augmentation.
        no_iterations (int): Total number of training iterations per task.
        mixup_epsilon_decay (str): Decaying rate of an epsilon in Interval MixUp.
        base_epsilon (float): Base perturbation value.
        current_epsilon (float): Current perturbation value, scheduled during training.
        current_kappa (float): Current kappa value, scheduled during training.
        regularization_targets (Any): Targets for regularization, set for tasks > 0.
        criterion (nn.Module): Loss function used for classification.
        current_iteration (int): Tracks the current training iteration within a task. 
            This value is incremented at each forward pass during training and is used to 
            schedule the values of epsilon and kappa for the SHIELD method. It is reset to 0 
            at the start of each new task via the setup_task method.
    Methods:
        schedule_epsilon(iteration: int) -> None:
        schedule_kappa(iteration: int) -> float:
        setup_task(task_id: int) -> None:
            Prepares the model for a new task, setting regularization targets and resetting schedules.
        forward(x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs a forward pass with mixup augmentation, scheduled perturbation, and regularization.
    """
   

    def __init__(self,
                 module: CLModuleABC,
                 epsilon: float,
                 lr: float,
                 use_lr_scheduler: bool,
                 beta: float,
                 mixup_alpha: float,
                 mixup_epsilon_decay: str = "linear",
                 final_kappa: float = 0.5
                 ):
        super().__init__(module=module, lr=lr, use_lr_scheduler=use_lr_scheduler)

        self.beta = beta
        self.mixup_alpha = mixup_alpha
        self.no_iterations = None

        self.base_epsilon = epsilon
        self.current_epsilon = 0.0
        self.current_kappa = 1.0

        self.regularization_targets = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_iteration = 0

        self.mixup_epsilon_decay_fnc = MixupEpsilonDecayRate(mixup_epsilon_decay)
        self.final_kappa = final_kappa

    def set_no_iterations(self, value: int) -> None:
        self.no_iterations = value

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
        The kappa decreases linearly until it reaches `final_kappa`.
        """
        if iteration <= self.no_iterations // 2:
            self.current_kappa = 1 - (iteration / self.no_iterations)
        else:
            self.current_kappa = self.final_kappa

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

        self.setup_optim()

        if task_id > 0:
            self.regularization_targets = hreg.get_current_targets(
                task_id, deepcopy(self.module.hnet)
            )

        self.current_epsilon = 0.0
        self.current_kappa = 1.0
        self.current_iteration = 0
        
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass for the SHIELD method with mixup data augmentation and regularization.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            y (torch.Tensor): Ground truth labels tensor of shape (batch_size,).
            task_id (int): Identifier for the current task, used for conditional hypernetwork weights and regularization.
        Returns:
            Tuple[torch.Tensor,torch.Tensor]: The computed loss for the current batch and worst-case predictions of the model.
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
            self.schedule_epsilon(self.current_iteration)
            self.schedule_kappa(self.current_iteration)

            self.current_iteration += 1

        # Apply mixup augmentation
        mixup_tensor_input, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
        eps_transformed = self.mixup_epsilon_decay_fnc(lam) * self.current_epsilon

        # Calculate predictions
        prediction, eps_prediction = self.module.forward(
            x=mixup_tensor_input,
            epsilon=eps_transformed, 
            task_id=task_id
        )

        z_lower = prediction - eps_prediction
        z_upper = prediction + eps_prediction
        z = torch.where((nn.functional.one_hot(y_a, prediction.size(-1))).bool(), z_lower, z_upper)

        loss_spec = self.mixup_criterion(self.criterion, z, y_a, y_b, lam)
        loss_fit  = self.mixup_criterion(self.criterion, prediction, y_a, y_b, lam)

        loss_current_task = self.current_kappa * loss_fit + (1 - self.current_kappa) * loss_spec

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

        # z_eval — used only for tracking worst-case predictions
        z_eval = torch.where((nn.functional.one_hot(y, prediction.size(-1))).bool(), z_lower, z_upper)

        return loss, z_eval
            
    def mixup_criterion(self, criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Computes the mixup loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def detach_embedding(self, task_id) -> None:
        """
        Detaches the `task_id`-th embedding from a computational graph.
        """
        self.module.hnet.conditional_params[task_id].requires_grad_(False)