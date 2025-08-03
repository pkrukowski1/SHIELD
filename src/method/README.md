## `method`

This module contains the core implementation of the SHIELD method and its supporting components.

- `interval_mixup_decay_rate.py` – Implements various decay schedules for the perturbation value used in Interval MixUp during training.
- `method_abc.py` – Abstract base class defining the common interface and structure for all continual learning methods.
- `shield.py` – Main implementation of the SHIELD method, including training logic, inference, and loss computation.
- `utils.py` – Utility functions supporting SHIELD’s training and evaluation pipeline (e.g., mixup virtual samples generation).
