import torch
from torch import nn
import hypnettorch.utils.hnet_regularizer as hreg

from method.method_abc import MethodABC

from typing import Tuple
from copy import deepcopy

class HNET(MethodABC):
   

    def __init__(self, 
                 beta: float,
                 ):
        """
        Initializes the hypernetwork module for continual learning (CL).
        This constructor sets up the hypernetwork with the specified beta parameter, 
        initializes the regularization targets, and defines the loss criterion as cross-entropy loss.

        Args:
            beta (float): Regularization strength or scaling parameter for the hypernetwork.
        """
        
        super().__init__()

        self.beta = beta
        
        self.regularization_targets = None
        self.criterion = nn.CrossEntropyLoss()

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
        """
        if task_id > 0:
            self.hnet.conditional_params[task_id-1].requires_grad_(False)

            self.regularization_targets = hreg.get_current_targets(
                task_id, deepcopy(self.module.hnet)
            )


    
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model, computes the loss for the current task, and adds regularization if applicable.
        Args:
            x (torch.Tensor): Input tensor for the model.
            y (torch.Tensor): Target tensor for the current task.
            task_id (int): Identifier for the current task.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The total loss tensor and predictions of the model.
        """


        # Calculate predictions
        prediction, _ = self.module.forward(
            x=x,
            epsilon=0.0, 
            task_id=task_id
        )

        loss_current_task  = self.criterion(prediction, y)

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
        
        return loss, prediction