import torch.nn as nn
import torch

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Computes the mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def calculate_worst_case_loss(z_lower: torch.Tensor, z_upper: torch.Tensor, 
                              prediction: torch.Tensor, gt_output: torch.Tensor) -> torch.Tensor:
        """
        Calculates the worst-case loss by selecting between lower and upper bounds for each class,
        based on the ground truth output.
            z_lower (torch.Tensor): The lower bound tensor for each class.
            z_upper (torch.Tensor): The upper bound tensor for each class.
            prediction (torch.Tensor): The model's predicted logits.
            gt_output (torch.Tensor): The ground truth class indices.
            torch.Tensor: A tensor where, for each sample, the value from z_lower is selected for the
                            ground truth class and z_upper for all other classes, representing the
                            worst-case scenario for loss computation.
        """
       
        z = torch.where((nn.functional.one_hot(gt_output, prediction.size(-1))).bool(), z_lower, z_upper)
        return z