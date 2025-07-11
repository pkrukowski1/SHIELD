from model.target_network.resnet18 import IntervalResNet18
from model.model_abc import CLModuleABC

from typing import Tuple

import torch
import torch.nn as nn
from hypnettorch.hnets import HMLP


class HyperNetWithResNet18(CLModuleABC):
    """
    A hypernetwork wrapper that generates the parameters of an AlexNet-like 
    target network using a conditional hypernetwork (HMLP).

    This class is designed for continual learning setups, where a separate 
    parameterization of the target network is learned for each task using 
    a hypernetwork conditioned on task embeddings.

    Attributes:
        target_network (IntervalResNet18): The target network that receives weights 
            from the hypernetwork.
        hnet (HMLP): Hypernetwork that generates the weights of the target network 
            conditioned on the task.
    """

    def __init__(self, 
                 in_shape: Tuple[int, int, int],
                 no_classes_per_task: int,
                 activation_function: nn.Module,
                 hnet_hidden_layers: Tuple[int, ...],
                 num_feature_maps: Tuple[int,int,int,int],
                 number_of_tasks: int,
                 hnet_embedding_size: int) -> None:
        """
        Initialize the HyperNetWithAlexNet module.

        Args:
            in_shape (Tuple[int, int, int]): Shape of the input images (C, H, W).
            no_classes_per_task (int): Number of output classes per task.
            activation_function (nn.Module): Activation function used in the hypernetwork.
            hnet_hidden_layers (Tuple[int, ...]): Sizes of the hidden layers in the hypernetwork.
            num_feature_maps (Tuple[int,int,int,int]): Number of feature maps for each ResNet block group,
              typically corresponding to the number of output channels in each convolutional block.
            number_of_tasks (int): Total number of tasks for continual learning.
            hnet_embedding_size (int): Size of the embedding used to condition the hypernetwork.
        """
        super().__init__()

        self.target_network = IntervalResNet18(
                in_shape=in_shape,
                use_bias=False,
                use_fc_bias=True,
                bottleneck_blocks=False,
                num_classes=no_classes_per_task,
                num_feature_maps=num_feature_maps,
                blocks_per_group=[2, 2, 2, 2],
                no_weights=True,
                use_batch_norm=True,
                projection_shortcut=True,
                bn_track_stats=False,
            )

        self.hnet = HMLP(
            self.target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=hnet_embedding_size,
            activation_fn=activation_function,
            layers=hnet_hidden_layers,
            num_cond_embs=number_of_tasks,
        )
        
    def forward(self, x: torch.Tensor, epsilon: float, task_id: int) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Perform a forward pass through the target network using weights generated 
        by the hypernetwork conditioned on the given task ID.

        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).
            epsilion (float): Perturbation value.
            task_id (int): ID of the current task, used to condition the hypernetwork.

        Returns:
            Tuple[torch.Tensor, float]: 
                - Output logits from the target network.
                - Output radii from the target_network.
        """
        hnet_weights = self.hnet.forward(cond_id=task_id)
        outputs, eps = self.target_network(
            x, epsilon=epsilon, weights=hnet_weights, condition=task_id
        )
        return outputs, eps
