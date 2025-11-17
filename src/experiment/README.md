## `experiment`

This module contains scripts for running key experiments and analyses used in the SHIELD framework.

- `adversarial_attack.py` – Evaluates model robustness under standard adversarial attacks such as FGSM, PGD, and AutoAttack.
- `calc_ver_accuracy.py` – Computes verified accuracy metrics based on interval bounds.
- `cil_adversarial_attack.py` – Performs adversarial evaluation in the Class-Incremental Learning (CIL) setting.
- `inc_no_iterations_pgd.py` – Analyzes the effect of increasing the number of PGD attack iterations on model performance.
- `inc_pert_size_fgsm.py` – Evaluates model robustness under varying perturbation sizes in FGSM attacks.
- `interval_training.py` – Main script for training models.
- `plot_interval_mixup_samples.py` – Visualizes interpolated and adversarial samples generated during interval-based training.
- `check_theorem_assumptions.py` – Checks whether the necessary conditions for maintaining robustness across continual learning tasks are satisfied.