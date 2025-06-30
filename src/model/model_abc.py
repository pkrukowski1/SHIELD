from abc import ABCMeta, abstractmethod

from torch import nn


class CLModuleABC(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for continual learning modules.

    This class serves as a base for all modules that require a set of learnable
    parameters and a defined forward pass. It enforces the implementation of the
    forward method in subclasses.

    Attributes:
        learnable_params (nn.ParameterList): A list of learnable parameters 
            specific to the module.

    Methods:
        forward(*args, **kwargs):
            Abstract method to be implemented by subclasses. Defines the forward 
            computation of the module.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CLModuleABC.

        Args:
            *args: Additional positional arguments for the nn.Module constructor.
            **kwargs: Additional keyword arguments for the nn.Module constructor.
        """
        super().__init__(*args, **kwargs)
        self._learnable_params = None

    @property
    def learnable_params(self):
        return self._learnable_params

    @learnable_params.setter
    def learnable_params(self, params):
        self._learnable_params = params

    @abstractmethod
    def forward(self, x, epsilon, task_id, *args, **kwargs):
        """
        Forward pass of the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_shape).
            epsilon (float): Perturbation value.
            task_id (int): Task identifier.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            Output of the forward pass.
        """
        pass