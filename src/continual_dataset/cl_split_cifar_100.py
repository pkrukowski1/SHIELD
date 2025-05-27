import numpy as np
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from typing import List

from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class SplitCIFAR100(ContinualLearningTaskGenerator):
    """
    Task generator for Split CIFAR-100.

    This dataset consists of (by default) 10 tasks, each with 10 disjoint classes from CIFAR-100.
    """

    def __init__(
        self,
        number_of_tasks: int = 10,
        seed: int = 42,
        use_augmentation: bool = False,
        use_cutout: bool = False,
    ):
        """
        Initialize the SplitCIFAR100.

        Args:
            number_of_tasks (int): Number of tasks (default 10 for 10x10 split).
            seed (int): Random seed for reproducibility (default 42).
            use_augmentation (bool): Whether to use data augmentation.
            use_cutout (bool): Whether to apply cutout (if supported).
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)
        self.use_augmentation = use_augmentation
        self.use_cutout = use_cutout
        self.no_classes_per_task = 100 // number_of_tasks

    def _generate_task_variations(self) -> List[range]:
        """
        Generate label splits for each task (e.g., [0-9], [10-19], ..., [90-99]).

        Returns:
            List[range]: A list of class label ranges for each task.
        """
        rng = np.random.default_rng(self.seed)
        class_permutation = rng.permutation(100)
        return [class_permutation[i:i + self.no_classes_per_task] 
                for i in range(0, 5 * self.number_of_tasks, self.no_classes_per_task)]

    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 0):
        """
        Prepare Split CIFAR-100 tasks.

        Args:
            datasets_folder (str): Folder for dataset download/storage.
            padding (int): Not used here, included for compatibility.
            validation_size (int): Number of validation samples per task.

        Returns:
            List[SplitCIFAR100Data]: List of dataset handlers for each task.
        """
        label_splits = self._generate_task_variations()
        handlers = [
            SplitCIFAR100Data(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=self.use_augmentation,
                use_cutout=self.use_cutout,
                labels=label_range,
            )
            for label_range in label_splits
        ]
        return handlers
