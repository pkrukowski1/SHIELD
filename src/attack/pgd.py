"""
The implementation is based on: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD
"""

import torch
import torch.nn as nn

from typing import Union

class PGD:
    """
    Implements the Projected Gradient Descent (PGD) attack for generating adversarial examples.

    Args:
        model (nn.Module): The neural network model to attack.
        task_id (int): Identifier for the task (used for multi-task models).
        eps (float, optional): Maximum perturbation allowed (L-infinity norm). Default: 8/255.
        alpha (float, optional): Step size for each attack iteration. Default: 2/255.
        steps (int, optional): Number of attack iterations. Default: 10.
        random_start (bool, optional): If True, initializes perturbation randomly within epsilon ball. Default: True.
        device (Union[int, torch.device], optional): Device to perform computations on. Default: "cpu".

    Example:
        >>> attack = PGD(model, task_id=0, eps=8/255, alpha=2/255, steps=10, random_start=True)
        >>> adv_images = attack.forward(images, labels)
    """

    def __init__(self, model: nn.Module, task_id: int, eps: float=8/255, alpha: float=2/255,
                  steps: int=10, random_start: bool=True, device: Union[int,torch.device]="cpu") -> None:
        super().__init__()
        self.model = model
        self.task_id = task_id
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the PGD (Projected Gradient Descent) attack to generate adversarial examples.
        Args:
            images (torch.Tensor): The input images to be perturbed. Shape: (batch_size, channels, height, width).
            labels (torch.Tensor): The true labels corresponding to the input images. Shape: (batch_size,).
        Returns:
            torch.Tensor: The adversarially perturbed images. Shape: (batch_size, channels, height, width).
        Notes:
            - If `self.targeted` is True, the attack aims to generate adversarial examples that are classified as the target labels.
            - If `self.random_start` is True, the attack starts from a random perturbation within the epsilon ball.
            - The attack iteratively updates the adversarial images using the sign of the gradient of the loss with respect to the input.
            - The perturbations are clipped to ensure they remain within the epsilon ball and valid image range [0, 1].
        """
       
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images, task_id=self.task_id, epsilon=0.0)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

