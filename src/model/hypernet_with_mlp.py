from model.target_network.mlp import IntervalMLP
from model.model_abc import CLModuleABC
from typing import Tuple

import torch
import torch.nn as nn
from hypnettorch.hnets import HMLP


class HyperNetWithMLP(CLModuleABC):
    """
    A hypernetwork wrapper that generates parameters for a multi-layer perceptron (MLP)
    target network using a conditional hypernetwork (HMLP).

    This setup is typically used in continual learning scenarios, where the
    hypernetwork outputs task-specific weights for the MLP based on a task ID embedding.

    Attributes:
        target_network (IntervalMLP): The main task-specific network that takes weights
            from the hypernetwork and processes inputs accordingly.
        hnet (HMLP): A conditional hypernetwork that generates weights for the MLP.
    """

    def __init__(self, 
                 in_shape: int,
                 no_classes_per_task: int,
                 activation_function: nn.Module,
                 hnet_hidden_layers: Tuple[int, ...],
                 target_hidden_layers: Tuple[int, ...],
                 number_of_tasks: int,
                 hnet_embedding_size: int) -> None:
        """
        Initialize the HyperNetWithMLP module.

        Args:
            in_shape (int): Number of input features (flattened image or vector input).
            no_classes_per_task (int): Number of output classes per task.
            activation_function (nn.Module): Activation function used in both MLP and hypernetwork.
            hnet_hidden_layers (Tuple[int, ...]): Sizes of hidden layers in the hypernetwork.
            target_hidden_layers (Tuple[int, ...]): Sizes of hidden layers in the MLP target network.
            number_of_tasks (int): Number of distinct tasks (used for conditional embeddings).
            hnet_embedding_size (int): Dimensionality of task embeddings in the hypernetwork.
        """

        super().__init__()

        self.target_network = IntervalMLP(
            n_in=in_shape,
            n_out=no_classes_per_task,
            hidden_layers=target_hidden_layers,
            activation_fn=activation_function,
            use_bias=True,
            no_weights=True,
        )

        self.hnet = HMLP(
            self.target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=hnet_embedding_size,
            activation_fn=activation_function,
            layers=hnet_hidden_layers,
            num_cond_embs=number_of_tasks,
        )
        

    def forward(self, x: torch.Tensor, task_id: int, epsilon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the target network using weights generated 
        by the hypernetwork conditioned on the given task ID.

        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).
            epsilon (float): Perturbation value.
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
