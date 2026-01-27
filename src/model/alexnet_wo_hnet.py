from typing import Tuple

import torch.nn as nn
import torch

from model.target_network.alexnet import IntervalAlexNet
from model.model_abc import CLModuleABC

class IntervalAlexNetWithoutHNET(CLModuleABC):
    """
    A standalone AlexNet wrapper that supports Interval Bound Propagation (IBP).
    
    This class wraps the stateless `IntervalAlexNet` functional network but manages 
    its own parameters (weights and biases) internally. This allows it to be used 
    as a standard PyTorch module without requiring a Hypernetwork.

    It automatically handles:
    1.  Parameter creation based on the functional network's shape requirements.
    2.  Proper initialization (Kaiming for weights, 1.0 for BN scales, 0.0 for biases).
    3.  Input reshaping (from flattened vectors to image tensors).
    4.  Interval propagation during the forward pass.

    Attributes:
        input_dims (Tuple[int, int, int]): Expected input image dimensions (C, H, W).
        functional_net (IntervalAlexNet): The stateless network that performs the operations.
        params (nn.ParameterList): The list of learnable parameters passed to the functional net.
    """
    def __init__(self, 
                 in_shape: Tuple[int, int, int],
                 no_classes_per_task: int,
                 activation_function: nn.Module,
                 ) -> None:
        """
        Initialize the IntervalAlexNet wrapper.

        Args:
            in_shape (Tuple[int, int, int]): The input image shape in (Channels, Height, Width).
            no_classes_per_task (int): The number of output classes (logits).
            activation_function (nn.Module): The activation function used (determines initialization gain).
        """
       
        super().__init__()

        self.input_dims = in_shape 

        self.functional_net = IntervalAlexNet(
            in_shape=in_shape,
            num_classes=no_classes_per_task,
            no_weights=True,
            use_batch_norm=True,
            bn_track_stats=False,
            distill_bn_stats=False
        )
                
        self.params = nn.ParameterList()
        
        for shape in self.functional_net.param_shapes:
            p = nn.Parameter(torch.empty(shape))
            self.params.append(p)

        self._initialize_weights(activation_function)

    def _initialize_weights(self, activation_fn: nn.Module):
        """
        Initialize the manually created parameters.

        This method applies Kaiming Uniform initialization to convolutional and linear weights,
        and specific initialization for Batch Normalization parameters (Scale=1, Shift=0).

        Args:
            activation_fn (nn.Module): The activation function module (e.g., nn.ReLU) used to 
                                       calculate the correct gain for Kaiming initialization.
        """
        nonlinearity = 'relu'
        if isinstance(activation_fn, nn.LeakyReLU):
            nonlinearity = 'leaky_relu'
        
        bn_start_idx = getattr(self.functional_net, "_bn_params_start_idx", len(self.params))

        for i, p in enumerate(self.params):
            
            if i < bn_start_idx:
                if p.dim() > 1:
                    torch.nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity=nonlinearity)
                else:
                    torch.nn.init.zeros_(p)
            
            else:
                relative_idx = i - bn_start_idx
                
                if relative_idx % 2 == 0:
                    torch.nn.init.ones_(p)
                else:
                    torch.nn.init.zeros_(p)

    def forward(self, 
                x: torch.Tensor, 
                task_id: int, 
                epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the manually created parameters.

        This method applies Kaiming Uniform initialization to convolutional and linear weights,
        and specific initialization for Batch Normalization parameters (Scale=1, Shift=0).

        Args:
            activation_fn (nn.Module): The activation function module (e.g., nn.ReLU) used to 
                                       calculate the correct gain for Kaiming initialization.
        """
       
        if x.dim() == 2:
            B = x.shape[0]
            if len(self.input_dims) == 3:
                C, H, W = self.input_dims
            else:
                C, H, W = 3, 32, 32
            x = x.view(B, C, H, W)

        outputs, eps = self.functional_net(
            x, 
            epsilon=epsilon, 
            weights=self.params,
            condition=task_id
        )
        
        return outputs, eps