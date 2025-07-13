import numpy as np
from hypnettorch.data.special import permuted_mnist

from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator

from typing import List

class PermutedMNIST(ContinualLearningTaskGenerator):
    """
    A utility class to generate a list of PermutedMNIST tasks for continual learning.

    Each task corresponds to a unique permutation of pixel indices applied to the MNIST dataset,
    which helps simulate different tasks in a continual learning setup.
    """

    def __init__(self, number_of_tasks: int, padding: int = 2, validation_size: int = 5000) -> None:
        """
        Initialize the PermutedMNISTTaskGenerator.

        Args:
            number_of_tasks (int, optional): The total number of rotated tasks.
            padding (int, optional): Amount of zero-padding to apply to each image. Defaults to 2.
            validation_size (int, optional): Number of samples to use for the validation set. Defaults to 5000.

        Attributes:
            input_shape (int): The number of pixels (e.g., 784 for 28x28 MNIST images).
            number_of_tasks (int): The total number of rotated tasks.
            seed (int, optional): Random seed for reproducibility (default 1).
            padding (int): Amount of zero-padding applied to each image.
            validation_size (int, optional): Number of samples to use for the validation set. Defaults to 5000.
        """
        super().__init__()
 
        self.number_of_tasks = number_of_tasks
        self.seed = 1
        self.padding = padding
        self.validation_size = validation_size

        self.input_shape = (28 + 2*self.padding)**2

    def _generate_task_variations(self) -> List[np.ndarray]:
        """
        Generate random permutations for each task.

        Returns:
            List[np.ndarray]: A list of permutation arrays.
        """
        np.random.seed(self.seed)
        return [np.random.permutation(self.input_shape) for _ in range(self.number_of_tasks)]

    def prepare_tasks(self, datasets_folder: str) -> permuted_mnist.PermutedMNISTList:
        """
        Create a list of PermutedMNIST tasks using the generated permutations.

        Args:
            datasets_folder (str): Path to store or load the MNIST dataset.

        Returns:
            PermutedMNISTList: A list-like object containing PermutedMNIST tasks.
        """
        permutations = self._generate_task_variations()
        return permuted_mnist.PermutedMNISTList(
            permutations=permutations,
            data_path=datasets_folder,
            use_one_hot=True,
            padding=self.padding,
            validation_size=self.validation_size,
        )
