from typing import List
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers

from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator

class SplitMNIST(ContinualLearningTaskGenerator):
    """
    Task generator for SplitMNIST.

    This version splits the MNIST dataset into `number_of_tasks` tasks,
    each with 2 consecutive digit classes (e.g., [0,1], [2,3], ..., [8,9]).
    """

    def __init__(
        self,
        number_of_tasks: int = 5,
        use_augmentation: bool = False,
        seed: int = None,
    ):
        """
        Initialize the SplitMNISTTaskGenerator.

        Args:
            number_of_tasks (int): Number of tasks (default: 5).
            use_augmentation (bool): Whether to use data augmentation.
            seed (int, optional): Unused, but included for interface compatibility.
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)
        self.use_augmentation = use_augmentation

    def _generate_task_variations(self):
        """
        Not used in this generator, as split is handled internally by `get_split_mnist_handlers`.

        Returns:
            None
        """
        return None

    def prepare_tasks(
        self, datasets_folder: str, padding: int = 0, validation_size: int = 0
    ) -> List:
        """
        Prepare SplitMNIST tasks.

        Args:
            datasets_folder (str): Directory for dataset download/storage.
            padding (int): Not used in this generator, kept for consistency.
            validation_size (int): Number of validation samples per task.

        Returns:
            List: A list of dataset handlers for each task.
        """
        return get_split_mnist_handlers(
            datasets_folder,
            use_one_hot=True,
            validation_size=validation_size,
            num_classes_per_task=2,
            num_tasks=self.number_of_tasks,
            use_torch_augmentation=self.use_augmentation,
        )
