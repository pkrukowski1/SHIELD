import torch
from torch import nn
import hypnettorch.utils.hnet_regularizer as hreg

from method.method_abc import MethodABC
from method.utils import mixup_data
from model.model_abc import CLModuleABC
from method.interval_mixup_decay_rate import MixupEpsilonDecayRate

from typing import Tuple
from copy import deepcopy
from enum import Enum

class ScenarioType(Enum):
    """Enumeration for the adversarial curriculum scenario."""
    PGD_FGSM_NONE = "pgd_fgsm_none"
    NONE_FGSM_PGD = "none_fgsm_pgd"

class AttackType(Enum):
    """Enumeration for the type of adversarial attack."""
    NONE = "none"
    FGSM = "fgsm"
    PGD = "pgd"

class SHIELDWithAdvAttacks(MethodABC):
    """
    SHIELD with Task-Based Adversarial Curriculum.

    This class extends the interval-based SHIELD method by incorporating a scheduled
    regimen of adversarial attacks (FGSM and PGD) to improve robustness. It adapts 
    the adversarial attack type based on the current task index and training progress.

    Args:
        module (CLModuleABC): The continual learning module (target net + hypernetwork).
        epsilon (float): Base perturbation radius for interval bound propagation.
        lr (float): Learning rate.
        use_lr_scheduler (bool): Whether to use a learning rate scheduler.
        beta (float): Hyperparameter scaling the regularization strength for previous tasks.
        mixup_alpha (float): Alpha parameter for the Beta distribution in MixUp.
        number_of_tasks (int): The total number of tasks in the continuum.
        mixup_epsilon_decay (str, optional): Strategy to decay epsilon during MixUp. Defaults to "linear".
        final_kappa (float, optional): The final weighting term for the fit loss. Defaults to 0.5.
        scenario (str, optional): The name of the attack curriculum (e.g., 'pgd_fgsm_none').
        pgd_steps (int, optional): Number of steps for the PGD attack. Defaults to 10.
        pgd_alpha (float, optional): Step size for the PGD attack. Defaults to 2/255.0.
        adv_epsilon (float, optional): Maximum perturbation radius for attacks. Defaults to 8/255.0.

    Attributes:
        scenario (ScenarioType): The active attack schedule scenario.
        enable_adv_training (bool): Flag that activates adversarial training (typically after warmup).
        current_epsilon (float): The current scheduled epsilon for Interval Training.
        current_kappa (float): The current scheduled kappa (balance between robustness and accuracy).
    """

    def __init__(self,
                 module: CLModuleABC,
                 epsilon: float,
                 lr: float,
                 use_lr_scheduler: bool,
                 beta: float,
                 mixup_alpha: float,
                 number_of_tasks: int,
                 mixup_epsilon_decay: str = "linear",
                 final_kappa: float = 0.5,
                 scenario: str = "pgd_fgsm_none",
                 pgd_steps: int = 10,
                 pgd_alpha: float = 2/255.0, 
                 adv_epsilon: float = 8/255.0
                 ):
        
        super().__init__(module=module, lr=lr, use_lr_scheduler=use_lr_scheduler)

        self.beta = beta
        self.mixup_alpha = mixup_alpha
        self.no_iterations = None
        self.number_of_tasks = number_of_tasks

        self.base_epsilon = epsilon
        self.current_epsilon = 0.0
        self.current_kappa = 1.0

        self.regularization_targets = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_iteration = 0

        self.mixup_epsilon_decay_fnc = MixupEpsilonDecayRate(mixup_epsilon_decay)
        self.final_kappa = final_kappa
        self.scenario = self._get_scenario(scenario)
        
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.adv_epsilon = adv_epsilon

        self.enable_adv_training = False

    def _get_scenario(self, scenario: str) -> ScenarioType:
        """Parses the scenario string into a ScenarioType enum."""
        if scenario == "pgd_fgsm_none":
            return ScenarioType.PGD_FGSM_NONE
        elif scenario == "none_fgsm_pgd":
            return ScenarioType.NONE_FGSM_PGD
        else:
            raise ValueError(f"Invalid SHIELD scenario: {scenario}")

    def set_no_iterations(self, value: int) -> None:
        """Sets the total number of iterations expected per task."""
        self.no_iterations = value

    def schedule_epsilon(self, iteration: int) -> None:
        """
        Schedules the epsilon value and enables adversarial training.

        The epsilon ramps up linearly during the first half of training. 
        Adversarial training is enabled only after the warmup phase (halfway point)
        to ensure the model has learned basic features first.
        """
        if iteration <= self.no_iterations // 2:
            self.current_epsilon = 2 * iteration / self.no_iterations * self.base_epsilon
        else:
            self.current_epsilon = self.base_epsilon
            self.enable_adv_training = True

    def schedule_kappa(self, iteration: int) -> float:
        """Schedules the kappa value (weighting between fit and spec loss)."""
        self.current_kappa = max(self.final_kappa, 1 - iteration / self.no_iterations)

    def setup_task(self, task_id: int) -> None:
        """
        Prepares the optimizer and regularization targets for a new task.

        Args:
            task_id (int): The identifier of the upcoming task.
        """
        self.setup_optim()

        if task_id > 0:
            self.regularization_targets = hreg.get_current_targets(
                task_id, deepcopy(self.module.hnet)
            )

        self.current_epsilon = 0.0
        self.current_kappa = 1.0
        self.current_iteration = 0
        self.enable_adv_training = False

    def _determine_attack_type(self, task_id: int) -> AttackType:
        """
        Determines which adversarial attack to apply based on the Task ID.

        Args:
            task_id (int): The current task identifier.

        Returns:
            AttackType: The type of attack (PGD, FGSM, or NONE).
        """
        is_first_task = (task_id == 0)
        is_last_task = (task_id == self.number_of_tasks - 1)
        
        if self.scenario == ScenarioType.PGD_FGSM_NONE:
            if is_first_task:
                return AttackType.PGD
            elif is_last_task:
                return AttackType.NONE
            else:
                return AttackType.FGSM

        elif self.scenario == ScenarioType.NONE_FGSM_PGD:
            if is_first_task:
                return AttackType.NONE
            elif is_last_task:
                return AttackType.PGD
            else:
                return AttackType.FGSM
                
        return AttackType.NONE

    def generate_adversarial_example(self, x: torch.Tensor, y: torch.Tensor, task_id: int, 
                                     attack_type: AttackType) -> torch.Tensor:
        """
        Generates an adversarial example using the specified attack method.

        Ensures valid image range [0, 1] and epsilon constraints.

        Args:
            x (torch.Tensor): Clean input images.
            y (torch.Tensor): Ground truth labels.
            task_id (int): Task identifier for the model condition.
            attack_type (AttackType): The method to use (FGSM or PGD).

        Returns:
            torch.Tensor: The adversarially perturbed images.
        """
        if attack_type == AttackType.NONE:
            return x

        self.module.eval()
        
        if attack_type == AttackType.FGSM:
            delta = torch.zeros_like(x)
        else:
            delta = torch.zeros_like(x).uniform_(-self.adv_epsilon, self.adv_epsilon)
            delta.data = torch.clamp(delta, -self.adv_epsilon, self.adv_epsilon)
            
        delta.requires_grad = True

        if attack_type == AttackType.FGSM:
            steps = 1
            alpha = self.adv_epsilon 
        else:
            steps = self.pgd_steps
            alpha = self.pgd_alpha

        for _ in range(steps):
            pred, _ = self.module.forward(x + delta, epsilon=0.0, task_id=task_id)
            loss = self.criterion(pred, y)
            loss.backward()

            with torch.no_grad():
                if delta.grad is not None:
                    delta.data += alpha * delta.grad.sign()
                    
                    delta.data = torch.clamp(delta, -self.adv_epsilon, self.adv_epsilon)
                    delta.data = torch.clamp(x + delta, 0.0, 1.0) - x
                    
                    delta.grad.zero_()

        self.module.train()
        return (x + delta).detach()

    def forward(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a training step with Interval Bound Propagation, MixUp, and optional Adversarial Training.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Label batch.
            task_id (int): Current task identifier.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The calculated loss and the worst-case prediction (z_eval).
        """
        if self.module.hnet.training:
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
            loss_regularization = hreg.calc_fix_target_reg(
                self.module.hnet,
                task_id,
                targets=self.regularization_targets,
                mnet=self.module.target_network,
                prev_theta=None,
                prev_task_embs=None,
                inds_of_out_heads=None,
                batch_size=-1,
            )

        loss = loss_current_task + self.beta * loss_regularization / max(1, task_id)

        # z_eval — used only for tracking worst-case predictions
        z_eval = torch.where((nn.functional.one_hot(y, prediction.size(-1))).bool(), z_lower, z_upper)

        if self.enable_adv_training and self.module.training:
            attack_type = self._determine_attack_type(task_id)

            if attack_type != AttackType.NONE:
                adv_x = self.generate_adversarial_example(x, y, task_id, attack_type)
                mixup_adv_x = lam * adv_x + (1 - lam) * adv_x[torch.randperm(len(adv_x))]

                adv_prediction, _ = self.module.forward(
                    x=mixup_adv_x,
                    epsilon=0.0, 
                    task_id=task_id
                )

                adv_loss_fit = self.mixup_criterion(self.criterion, adv_prediction, y_a, y_b, lam)
                loss = loss + self.current_kappa * adv_loss_fit

        return loss, z_eval
            
    def mixup_criterion(self, criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Computes the mixup loss linear combination."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def detach_embedding(self, task_id) -> None:
        """Detaches the `task_id`-th embedding from the computational graph."""
        self.module.hnet.conditional_params[task_id].requires_grad_(False)