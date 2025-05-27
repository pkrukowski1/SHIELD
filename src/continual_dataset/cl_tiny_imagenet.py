import numpy as np
from typing import List

from continual_dataset.dataset.tiny_imagenet import TinyImageNet
from  continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class TinyImageNet(ContinualLearningTaskGenerator):
    """
    Task generator for TinyImageNet.

    The dataset is split into `number_of_tasks`, each containing equal number of disjoint classes.
    """

    def __init__(
        self,
        number_of_tasks: int = 40,
        seed: int = 42
    ):
        """
        Initialize the TinyImageNet.

        Args:
            number_of_tasks (int): Total number of tasks (default 40).
            seed (int): Random seed for reproducibility (default 42).
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)
        self.no_classes_per_task = 200 // number_of_tasks

    def _generate_task_variations(self) -> List[np.ndarray]:
        """
        Generate a class permutation split into tasks.

        Returns:
            List[np.ndarray]: Each element contains 5 class indices for a task.
        """
        rng = np.random.default_rng(self.seed)
        class_permutation = rng.permutation(200)
        return [class_permutation[i:i + self.no_classes_per_task] 
                for i in range(0, 5 * self.number_of_tasks, self.no_classes_per_task)]

    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 250):
        """
        Prepare TinyImageNet tasks according to the WSN setup.

        Args:
            datasets_folder (str): Directory where the dataset is or will be stored.
            padding (int): Unused for TinyImageNet but kept for compatibility.
            validation_size (int): Number of validation samples per task (default 250).

        Returns:
            List[TinyImageNet]: List of TinyImageNet dataset handlers for each task.
        """
        task_labels = self._generate_task_variations()
        handlers = []

        for labels in task_labels:
            print(f"Order of classes in the current task: {labels}")
            handlers.append(
                TinyImageNet(
                    data_path=datasets_folder,
                    validation_size=validation_size,
                    use_one_hot=True,
                    labels=labels,
                )
            )

        return handlers
