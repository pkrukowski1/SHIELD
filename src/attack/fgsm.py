"""
The implementation is based on the following implementation: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/fgsm.html#FGSM
"""

import torch
import torch.nn as nn

from typing import Union

class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Reference:
        - "Explaining and harnessing adversarial examples" (https://arxiv.org/abs/1412.6572)

    Args:
        model (nn.Module): Model to attack.
        task_id (int): Task identifier for the model's forward pass.
        eps (float, optional): Maximum perturbation. Default: 8/255.
        device (str or torch.device, optional): Device to use. Default: "cpu".

    Inputs:
        images (torch.Tensor): Input images of shape (N, C, H, W), with values in [0, 1].
        labels (torch.Tensor): Ground truth labels of shape (N,).

    Returns:
        torch.Tensor: Adversarial images of shape (N, C, H, W).
    """

    def __init__(self, model: nn.Module, task_id: int, eps: float=8/255, device: Union[str,torch.device]="cpu") -> None:
        super().__init__()
        self.model = model
        self.task_id = task_id
        self.eps = eps
        self.device = device

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generates adversarial examples using the Fast Gradient Sign Method (FGSM).
        Args:
            images (torch.Tensor): The input images to perturb, of shape (N, C, H, W),
                where N is the batch size, C is the number of channels, H is the height,
                and W is the width.
            labels (torch.Tensor): The ground truth labels corresponding to the input images.
        Returns:
            torch.Tensor: The adversarially perturbed images, of the same shape as the input images.
        """
        

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs, _ = self.model(images, task_id=self.task_id, epsilon=0.0)

        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

