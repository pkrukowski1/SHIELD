import numpy as np
from typing import List

from continual_dataset.dataset.tiny_imagenet import TinyImageNetData
from  continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class TinyImageNet(ContinualLearningTaskGenerator):
    """
    Task generator for TinyImageNet.

    The dataset is split into `number_of_tasks`, each containing equal number of disjoint classes.
    """

    def __init__(
        self,
        number_of_tasks: int = 40,
        validation_size: int = 250,
        use_augmentation: bool = False
    ) -> None:
        """
        Initialize the TinyImageNet.

        Args:
            number_of_tasks (int): Total number of tasks (default 40).
            validation_size (int): Number of samples in a validation set.
            use_augmentation (bool): Flag to indicate wheter data augmentation should be used.
        """
        super().__init__()

        self.number_of_tasks = number_of_tasks
        self.validation_size = validation_size
        self.use_augmentation = use_augmentation
        self.no_classes_per_task = 200 // number_of_tasks

    def _generate_task_variations(self) -> None:
        """
        Not used in this generator, as split is handled internally.

        Returns:
            None
        """
        
        return None

    def prepare_tasks(self, datasets_folder: str) -> List[TinyImageNetData]:
        """
        Creates and returns a list of TinyImageNet dataset handlers, each corresponding to a specific task configuration.
        
        Args:
            datasets_folder (str): Path to the directory containing or intended to contain the TinyImageNet dataset.

        Returns:
            List[TinyImageNetData]: A list of TinyImageNet dataset handler instances, one for each generated task configuration.

        Notes:
            - The method generates different task configurations using internal logic and prepares a TinyImageNet handler for each.
            - Prints the order of class labels for each task for transparency and debugging.
            - The `validation_size` and other relevant parameters are taken from the instance attributes.
        """
        seed = 1
        rng = np.random.default_rng(seed)
        class_permutation = rng.permutation(200)

        task_labels = [class_permutation[i:i + self.no_classes_per_task] 
                for i in range(0, 5 * self.number_of_tasks, self.no_classes_per_task)]
        handlers = []

        for labels in task_labels:
            print(f"Order of classes in the current task: {labels}")
            handlers.append(
                TinyImageNetData(
                    data_path=datasets_folder,
                    use_one_hot=True,
                    use_data_augmentation=self.use_augmentation,
                    validation_size=self.validation_size,
                    seed=seed,
                    labels=labels,
                )
            )

        return handlers
