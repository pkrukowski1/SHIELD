import numpy as np
from typing import List

from continual_dataset.dataset.rotated_mnist import RotatedMNISTlist
from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class RotatedMNIST(ContinualLearningTaskGenerator):
    """
    A utility class to generate a list of RotatedMNIST tasks for continual learning.

    Each task corresponds to a unique random rotation applied to the MNIST images,
    helping simulate different tasks in a continual learning setup.
    """

    def __init__(self, number_of_tasks: int, padding: int = 2, validation_size: int = 5000) -> None:
        """
        Initializes the RotatedMNISTTaskGenerator.

        Args:
            number_of_tasks (int, optional): The total number of rotated tasks.
            padding (int, optional): Amount of zero-padding to apply to each image. Defaults to 2.
            validation_size (int, optional): Number of samples to use for the validation set. Defaults to 5000.

        Attributes:
            number_of_tasks (int): The total number of rotated tasks.
            seed (int): Random seed for reproducibility.
            padding (int): Amount of zero-padding applied to each image.
            validation_size (int): Number of samples in the validation set.
        """
        super().__init__()

        self.number_of_tasks = number_of_tasks
        self.seed = 1
        self.padding = padding
        self.validation_size = validation_size

    def _generate_task_variations(self) -> List[np.ndarray]:
        """
        Generate random rotations (in degrees) for each task.

        Returns:
            List[np.ndarray]: A list of rotation angles in degrees.
        """

        np.random.seed(self.seed)
        return [np.random.uniform(0, 360) for _ in range(self.number_of_tasks)]

    def prepare_tasks(self, datasets_folder: str) -> RotatedMNISTlist:
        """
        Create a list of RotatedMNIST tasks using the generated rotation angles.

        Args:
            datasets_folder (str): Path to store or load the MNIST dataset.

        Returns:
            RotatedMNISTlist: A RotatedMNISTlist object representing RotatedMNIST tasks.
        """
        rotations = self._generate_task_variations()
        return RotatedMNISTlist(
            angles=rotations,
            data_path=datasets_folder,
            use_one_hot=True,
            validation_size=self.validation_size,
            padding=self.padding
        )
