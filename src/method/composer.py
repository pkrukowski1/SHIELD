import logging
from typing import Optional
from copy import deepcopy

import torch
from torch import nn
from torch import optim

import wandb

from method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Composer:
    """
    Composer class for managing the training process of a module with optional plugins and regularization.

    Attributes:
        module (CLModuleABC): The module to be trained.
        optimizer (Optional[optim.Optimizer]): The optimizer for training.
        lr (float): The learning rate for subsequent tasks.
        plugins (Optional[list[MethodPluginABC]]): List of plugins to be used during training.
        criterion (nn.Module): The loss function used for training on a current task, defaults to CrossEntropyLoss.
    """

    def __init__(self, 
        module: nn.Module,
        lr: float,
        plugins: Optional[list[MethodPluginABC]]=[]
    ):
        """
        Initialize the Composer class.

        Args:
            module (CLModuleABC): The continual learning module to be trained.
            lr (float): The learning rate for subsequent tasks.
            ema_scale (float): Exponential moving average scale for dynamic loss scaling.
            plugins (Optional[list[MethodPluginABC]], optional): List of method plugins to extend the training process. Defaults to an empty list.
        """

        self.module = module

        self.optimizer = None
        self.lr = lr
        self.plugins = plugins
        self.criterion = nn.CrossEntropyLoss()

        for plugin in self.plugins:
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def _setup_optim(self, task_id: int):
        """
        Sets up the optimizer for the model.
        This method initializes the optimizer with the model parameters that require
        gradients. It uses the Adam optimizer with a learning rate that depends on
        the task ID. If the task ID is 0, it uses `first_lr`, otherwise it uses `lr`.

        Args:
            task_id (int): The ID of the current task. Determines the learning rate to use.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        self.optimizer = optim.Adam(params, lr=self.lr)


    def setup_task(self, task_id: int):
        """
        Set up the task with the given task ID.
        This method initializes the optimizer for the specified task and
        calls the setup_task method on each plugin associated with this instance.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        self._setup_optim(task_id)
        for plugin in self.plugins:
            plugin.setup_task(task_id)


    def forward(self, x, y, task_id):
        """
        Perform a forward pass through the model and apply plugins.

        Args:
            x (torch.Tensor): Input tensor to the model.
            y (torch.Tensor): Target tensor for computing the loss.
            task_id (int): The ID of the current task.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss after applying regularization and plugins.
                - preds (torch.Tensor): The model predictions after applying plugins.
        """

        preds = self.module(x, task_id)
        loss = self.criterion(preds, y)
    
        for plugin in self.plugins:
            loss, preds = plugin.forward(x, y, loss, preds)

        return loss, preds


    def backward(self, loss):  
        """
        Performs a backward pass and updates the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to prevent exploding gradients.
        4. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()