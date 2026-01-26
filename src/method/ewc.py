import torch
from torch import nn
import hypnettorch.utils.ewc_regularizer as ewc_reg

from method.method_abc import MethodABC
from method.utils import mixup_data
from model.model_abc import CLModuleABC
from method.interval_mixup_decay_rate import MixupEpsilonDecayRate

from typing import Tuple, Iterable
import numpy as np

class EWC(MethodABC):
    """
    Elastic Weight Consolidation (EWC) method with Interval Bound Propagation (IBP) and Mixup.

    This class implements a continual learning strategy that combines:
    1. EWC: Mitigates catastrophic forgetting by penalizing changes to parameters important for previous tasks.
    2. IBP (Interval Bound Propagation): Robustness training using interval arithmetic (epsilon perturbations).
    3. Mixup: Data augmentation technique blending inputs and labels.

    Attributes:
        alpha (float): Scaling factor for the EWC regularization term.
        mixup_alpha (float): Alpha parameter for the Beta distribution used in Mixup.
        base_epsilon (float): The maximum perturbation radius for interval bounds.
        current_epsilon (float): The current perturbation radius (scheduled during training).
        current_kappa (float): The current balancing factor between standard loss and specification loss.
        final_kappa (float): The minimum value for kappa at the end of training.
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        mixup_epsilon_decay_fnc (MixupEpsilonDecayRate): Strategy for decaying epsilon based on mixup lambda.
    """
    def __init__(self,
                 module: CLModuleABC,
                 epsilon: float,
                 lr: float,
                 alpha: float,
                 use_lr_scheduler: bool,
                 mixup_alpha: float,
                 mixup_epsilon_decay: str = "linear",
                 final_kappa: float = 0.5
                ):
        """
        Initialize the EWC method.

        Args:
            module (CLModuleABC): The continual learning module (network) to be trained.
            epsilon (float): The maximum radius for interval bound propagation (IBP).
            lr (float): Learning rate for the optimizer.
            alpha (float): Regularization strength for EWC.
            use_lr_scheduler (bool): Whether to use a learning rate scheduler.
            mixup_alpha (float): Concentration parameter for the Beta distribution in Mixup.
            mixup_epsilon_decay (str, optional): The decay strategy for epsilon during Mixup. 
                                                 Defaults to "linear".
            final_kappa (float, optional): The final value for the loss weighting factor kappa. 
                                           Defaults to 0.5.
        """
        super().__init__(module=module, lr=lr, use_lr_scheduler=use_lr_scheduler)

        self.alpha = alpha
        self.mixup_alpha = mixup_alpha
        self.no_iterations = None

        self.base_epsilon = epsilon
        self.current_epsilon = 0.0
        self.current_kappa = 1.0
    
        self.regularization_targets = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_iteration = 0

        self.mixup_epsilon_decay_fnc = MixupEpsilonDecayRate(mixup_epsilon_decay)
        self.final_kappa = final_kappa

    def set_no_iterations(self, value: int) -> None:
        """
        Sets the total number of iterations for the current task training.
        
        This value is required for scheduling `current_epsilon` and `current_kappa`
        over the course of training.

        Args:
            value (int): Total number of iterations (epochs * batches_per_epoch).
        """
        self.no_iterations = value

    def schedule_epsilon(self, iteration: int) -> None:
        """
        Schedules the perturbation epsilon based on the current iteration.
        
        The epsilon increases linearly from 0 to `base_epsilon` over the first half 
        of the training iterations to stabilize early training.

        Args:
            iteration (int): The current training iteration index.
        """
        if iteration <= self.no_iterations // 2:
            self.current_epsilon = 2 * iteration / self.no_iterations * self.base_epsilon
        else:
            self.current_epsilon = self.base_epsilon

    def schedule_kappa(self, iteration: int) -> float:
        """
        Schedules the kappa value based on the current iteration.
        
        Kappa decreases linearly from 1.0 to `final_kappa` over the course of training,
        shifting emphasis from standard accuracy (fit) to robust accuracy (spec).
        
        Args:
            iteration (int): The current training iteration index.
        """
        self.current_kappa = max(self.final_kappa, 1 - iteration / self.no_iterations)

    def setup_optim(self) -> None:
        """
        Sets up the optimizer and the learning rate scheduler.
        
        Initializes an Adam optimizer for all parameters requiring gradients.
        If `use_lr_scheduler` is True, initializes a `ReduceLROnPlateau` scheduler.
        """

        params = filter(lambda p: p.requires_grad, self.module.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "max",
                factor=np.sqrt(0.1),
                patience=5,
                min_lr=0.5e-6,
                cooldown=0,
                verbose=True,
            )

    def setup_task(self, task_id: int, task_dataset: Iterable = None) -> None:
        """
        Prepares the model for training on a specific task.

        If this is not the first task (task_id > 0), it computes the Fisher Information Matrix 
        (FIM) for the previous task to establish EWC regularization constraints.

        Args:
            task_id (int): The identifier of the new task.
            task_dataset (Iterable, optional): The dataset for the previous task (required 
                                               to compute Fisher Information).
        
        Side Effects:
            - Resets `current_epsilon` to 0.0 and `current_kappa` to 1.0.
            - Resets `current_iteration` counter.
            - Calls `ewc_reg.compute_fisher` if `task_id > 0`.
            - Re-initializes the optimizer via `setup_optim`.
        """

        self.current_epsilon = 0.0
        self.current_kappa = 1.0
        self.current_iteration = 0

        if task_id > 0:
            ewc_reg.compute_fisher(
                task_id=task_id,
                data=task_dataset,
                params=self.module.parameters(),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                mnet=self.module,
            )

        self.setup_optim()

    
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Performs a forward pass, calculates loss (Standard + IBP + EWC), and handles Mixup.

        This method:
        1. Updates schedules for epsilon and kappa (if training).
        2. Applies Mixup to inputs `x`.
        3. Propagates interval bounds to get logits and radii.
        4. Computes the combined loss:
           Loss = Kappa * Fit_Loss + (1-Kappa) * Spec_Loss + Alpha * EWC_Regularization.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Labels for the input batch.
            task_id (int): The current task identifier.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The total computed loss (scalar tensor).
                - The worst-case logits (z_eval) for evaluation/metrics.
        """
        if self.module.training:
            self.schedule_epsilon(self.current_iteration)
            self.schedule_kappa(self.current_iteration)

            self.current_iteration += 1

        # Apply mixup augmentation
        mixup_tensor_input, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
        eps_transformed = self.mixup_epsilon_decay_fnc(lam) * self.current_epsilon

        # Calculate predictions
        prediction, eps_prediction = self.module.forward(
            x=mixup_tensor_input,
            epsilon=eps_transformed, 
            task_id=task_id
        )

        z_lower = prediction - eps_prediction
        z_upper = prediction + eps_prediction
        z = torch.where((nn.functional.one_hot(y_a, prediction.size(-1))).bool(), z_lower, z_upper)

        loss_spec = self.mixup_criterion(self.criterion, z, y_a, y_b, lam)
        loss_fit  = self.mixup_criterion(self.criterion, prediction, y_a, y_b, lam)

        loss_current_task = self.current_kappa * loss_fit + (1 - self.current_kappa) * loss_spec

        loss_regularization = 0.0
        if task_id > 0:
            loss_regularization = ewc_reg.ewc_regularizer(
                task_id=task_id,
                params=self.module.parameters(),
                mnet=self.module,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        loss = loss_current_task + self.alpha * loss_regularization / max(1, task_id)

        z_eval = torch.where((nn.functional.one_hot(y, prediction.size(-1))).bool(), z_lower, z_upper)

        return loss, z_eval
            
    def mixup_criterion(self, criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Computes the mixup loss by blending losses for both target sets.

        Args:
            criterion (nn.Module): The base loss function (e.g., CrossEntropy).
            pred (torch.Tensor): Predictions (logits).
            y_a (torch.Tensor): First set of targets.
            y_b (torch.Tensor): Second set of targets (permuted).
            lam (float): Mixup interpolation coefficient.

        Returns:
            torch.Tensor: Weighted loss.
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
