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
        validation_size: int = 1000,
    ) -> None:
        """
        Initializes the SplitMNISTTaskGenerator.
        
        Args:
            number_of_tasks (int, optional): Number of tasks to split the MNIST dataset into. Defaults to 5.
            use_augmentation (bool, optional): If True, applies data augmentation to the dataset. Defaults to False.
            validation_size (int, optional): Number of samples to use for validation. Defaults to 1000.
        """
        super().__init__()

        self.number_of_tasks = number_of_tasks
        self.use_augmentation = use_augmentation
        self.validation_size = validation_size

    def _generate_task_variations(self) -> None:
        """
        Not used in this generator, as split is handled internally by `get_split_mnist_handlers`.

        Returns:
            None
        """
        return None

    def prepare_tasks(
        self, datasets_folder: str
    ) -> List:
        """
        Prepare SplitMNIST tasks.

        Args:
            datasets_folder (str): Directory for dataset download/storage.

        Returns:
            List: A list of dataset handlers for each task.
        """
        return get_split_mnist_handlers(
            datasets_folder,
            use_one_hot=True,
            validation_size=self.validation_size,
            num_classes_per_task=2,
            num_tasks=self.number_of_tasks,
            use_torch_augmentation=self.use_augmentation,
        )
