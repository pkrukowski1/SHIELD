from abc import ABCMeta, abstractmethod
from model.model_abc import CLModuleABC

import torch

import numpy as np
from typing import Tuple

class MethodABC(metaclass=ABCMeta):
    """
    Base class for continual learning methods.

    Methods:
        setup_task(task_id: int):
            Abstract method for setting up a task. Must be implemented by subclasses.
        forward(x, y, loss, preds):
            Abstract method for the forward pass. Must be implemented by subclasses.
        set_module(module: CLModuleABC):
            Set the module for the plugin.
    """

    def __init__(self, 
                 module: CLModuleABC, 
                 lr: float, 
                 use_lr_scheduler: bool = False) -> None:
        """
        Args:
            module(CLModuleABC): The model to be set.
            lr (float): Learning rate for the optimizer.
        """

        self.module = module
        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler = None

        self.setup_optim()


    @abstractmethod
    def setup_task(self, task_id: int) -> None:
        """
        Internal setup task. It is useful when some CL methods store
        additional piece of information per task.
        
        Args:
            task_id (int): The unique identifier of the task to be set up.
        """

        pass


    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass of the model. It should return loss function values
        and predictions of the model.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target data or labels.
            task_id (int): Identifier for the current task.

        Returns:
            The output of the forward computation:
            - Loss function value.
            - Predictions of the model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    def setup_optim(self) -> None:
        """
        Sets up the optimizer and the scheduler (if applicable) for the model.
        This method initializes the optimizer with the model parameters that require
        gradients.
        """

        params = list(self.module.learnable_params)

        # Double check if the parameters require gradients
        params = filter(lambda p: p.requires_grad, params)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "max",
                factor=np.sqrt(0.1),
                patience=5,
                min_lr=0.5e-6,
                cooldown=0,
                verbose=True,
            )


    def backward(self, loss: torch.Tensor) -> None:  
        """
        Performs a backward pass and updates the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_scheduler_step(self, val_accuracy: float) -> None:
        """
        Makes a step for the learning rate scheduler.

        Args:
            val_accuracy (float): The validation accuracy metric to be used by the learning rate scheduler
            for adjusting the learning rate. This value is typically obtained after evaluating the model
            on the validation dataset at the end of an epoch.
        This method calls the step function of the learning rate scheduler with the specified validation
        accuracy metric, allowing the scheduler to update the learning rate based on model performance.
        """
        
        if self.use_lr_scheduler:
            self.scheduler.step(val_accuracy)