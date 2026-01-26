from typing import Tuple
import torch
import torch.nn as nn

from model.target_network.mlp import IntervalMLP
from model.model_abc import CLModuleABC

class IntervalMLPWithoutHNET(CLModuleABC):
    """
    A classical MLP wrapper for a stateless IntervalMLP.
    
    Since IntervalMLP is set to `no_weights=True`, this class manually 
    initializes, stores, and manages the weights as nn.Parameters, 
    passing them to the functional network during the forward pass.
    """

    def __init__(self, 
                 in_shape: int,
                 no_classes_per_task: int,
                 activation_function: nn.Module,
                 hidden_layers: Tuple[int, ...],
                 use_bias: bool = True) -> None:
        """
        Args:
            in_shape (int): Input dimension.
            no_classes_per_task (int): Output dimension.
            activation_function (nn.Module): Activation function (e.g., nn.ReLU).
            hidden_layers (Tuple[int, ...]): Tuple of hidden layer sizes.
            use_bias (bool): Whether to include bias parameters.
        """
        super().__init__()

        self.functional_net = IntervalMLP(
            n_in=in_shape,
            n_out=no_classes_per_task,
            hidden_layers=hidden_layers,
            activation_fn=activation_function,
            use_bias=use_bias,
            no_weights=True
        )

        self.params = nn.ParameterList()
        
        for shape in self.functional_net.param_shapes:
            p = nn.Parameter(torch.empty(shape))
            self.params.append(p)

        self._initialize_weights(activation_function)

    def _initialize_weights(self, activation_fn):
        """
        Manually applies initialization since we aren't using nn.Linear.
        """
        gain = 1.0
        if isinstance(activation_fn, nn.ReLU):
            gain = torch.nn.init.calculate_gain('relu')
        elif isinstance(activation_fn, nn.LeakyReLU):
            gain = torch.nn.init.calculate_gain('leaky_relu')
        elif isinstance(activation_fn, nn.Tanh):
            gain = torch.nn.init.calculate_gain('tanh')
        
        for p in self.params:
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p, gain=gain)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, 
                x: torch.Tensor, 
                task_id: int, 
                epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the manually stored parameters to the stateless network.
        """
        outputs, eps = self.functional_net(
            x, 
            epsilon=epsilon, 
            weights=self.params,
            condition=task_id
        )
        
        return outputs, eps