import torch.nn as nn
import torch

from autoattack import AutoAttack

from typing import Union, Tuple

class AttackModelWrapper(nn.Module):
    """
    A wrapper class for neural network models to facilitate adversarial attacks.

    This class wraps a given model and its associated weights, providing a forward method
    that computes the model's logits with specific parameters suitable for attack scenarios.

    Args:
        model (nn.Module): The neural network model to be wrapped.
        task_id (int): Identifier for the specific task or condition to be used in the model's forward pass.
        total_no_classes (int): The total number of classes in the dataset, used for padding logits.
        device (torch.device or str): The device on which computations will be performed.

    Methods:
        forward(x):
            Performs a forward pass through the wrapped model with epsilon set to 0.0,
            using the provided weights and no additional condition. Returns the logits
            output by the model.

    Example:
        wrapper = AttackModelWrapper(model, weights, device)
        logits = wrapper(input_tensor)
    """
    def __init__(self, model: nn.Module, task_id: int, total_no_classes: int, 
                 device: Union[str,torch.device] = "cpu") -> None:
        super(AttackModelWrapper, self).__init__()
        self.model = model
        self.task_id = task_id
        self.total_no_classes = total_no_classes
        self.device = device

    def forward(self, x):
        logits, _ = self.model(x, task_id=self.task_id, epsilon=0.0)

        # Pad logits to full number of classes with very negative values
        num_classes = logits.size(1)
        if num_classes < self.total_no_classes:
            pad_size = self.total_no_classes - num_classes
            padding = torch.full((logits.size(0), pad_size), -1e10, device=logits.device)
            logits = torch.cat([logits, padding], dim=1)

        return logits

    
class AutoAttackWrapper(AutoAttack):
    """
    A wrapper class for the AutoAttack adversarial attack suite, providing a unified interface
    for evaluating model robustness across different datasets and input shapes.
    Args:
        model (torch.nn.Module): The neural network model to attack.
        weights (str or dict): Path to the model weights or a state dict.
        eps (float): Maximum perturbation allowed for the attack.
        input_shape (Tuple[int,int,int]): Expected input shape for the dataset (C, H, W).
        total_no_classes (int): Total number of classes in the dataset for padding logits.
        device (torch.device or str): Device to run the attack on.
        norm (str, optional): Norm to use for the attack ('Linf', 'L2', etc.). Default is 'Linf'.
        version (str, optional): Version of AutoAttack to use. Default is 'custom'.
    Attributes:
        model_wrapper (AttackModelWrapper): Wrapped model for attack compatibility.
        input_shape (tuple): Expected input shape for the dataset.
    Methods:
        forward(images, labels, task_id):
            Runs the standard AutoAttack evaluation on the provided images and labels.
            Args:
                images (torch.Tensor): Input images to attack.
                labels (torch.Tensor): True labels for the images.
                task_id (int): Task identifier (not used in attack).
            Returns:
                dict: Results of the adversarial evaluation.
    """
    def __init__(self, model: nn.Module, task_id: int, eps: float, input_shape: Tuple[int,int,int], 
                 total_no_classes: int, device: Union[torch.device,str], norm: str='Linf', version: str='custom') -> None:
        self.model_wrapper = AttackModelWrapper(model, task_id, total_no_classes, device)

        super(AutoAttackWrapper, self).__init__(
            model=self.model_wrapper,
            norm=norm,
            eps=eps,
            version=version,
            verbose=True,
            device=device,
            attacks_to_run=["apgd-ce", "apgd-t", "fab", "square"]
        )
        self.model_wrapper.to(device)
        self.input_shape = input_shape
        

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Runs the standard AutoAttack evaluation on the provided images and labels.
        Args:
            images (torch.Tensor): Input images to attack, expected shape (N, C, H, W).
            labels (torch.Tensor): True labels for the images, expected shape (N,).
        Returns:
            torch.Tensor: Results of the adversarial evaluation.
        """
        images = images.view(images.shape[0], *self.input_shape)
        return self.run_standard_evaluation(images, labels)
