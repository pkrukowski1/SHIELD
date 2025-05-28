from abc import ABCMeta, abstractmethod

from model.model_abc import CLModuleABC

class MethodABC(metaclass=ABCMeta):
    """
    Base class for continual learning methods as plugins for composer.

    Methods:
        setup_task(task_id: int):
            Abstract method for setting up a task. Must be implemented by subclasses.
        forward(x, y, loss, preds):
            Abstract method for the forward pass. Must be implemented by subclasses.
        set_module(module: CLModuleABC):
            Set the module for the plugin.
    """

    def set_module(self, module: CLModuleABC):
        """
        Set the module for the plugin.
        
        Args:
            module(CLModuleABC): The model to be set.
        """

        self.module = module


    @abstractmethod
    def setup_task(self, task_id: int):
        """
        Internal setup task.
        
        Args:
            task_id (int): The unique identifier of the task to be set up.
        """

        pass


    @abstractmethod
    def forward(self, x, y, loss, preds):
        """
        Internal forward pass.
        """

        pass