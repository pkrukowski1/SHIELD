import numpy as np
from hypnettorch.data.special import permuted_mnist

from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator

class PermutedMNIST(ContinualLearningTaskGenerator):
    """
    A utility class to generate a list of PermutedMNIST tasks for continual learning.

    Each task corresponds to a unique permutation of pixel indices applied to the MNIST dataset,
    which helps simulate different tasks in a continual learning setup.
    """

    def __init__(self, input_shape: int, number_of_tasks: int, seed: int = 42):
        """
        Initialize the PermutedMNISTTaskGenerator.

        Args:
            input_shape (int): The number of pixels (e.g., 784 for 28x28 MNIST images).
            number_of_tasks (int): The total number of permuted tasks to generate.
            seed (int): Random seed for reproducibility (default 42).
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)
        self.input_shape = input_shape

    def _generate_task_variations(self):
        """
        Generate random permutations for each task.

        Returns:
            List[np.ndarray]: A list of permutation arrays.
        """
        return [np.random.permutation(self.input_shape) for _ in range(self.number_of_tasks)]

    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 0):
        """
        Create a list of PermutedMNIST tasks using the generated permutations.

        Args:
            datasets_folder (str): Path to store or load the MNIST dataset.
            padding (int, optional): Padding to apply to each image. Defaults to 0.
            validation_size (int, optional): Number of validation samples per task. Defaults to 0.

        Returns:
            PermutedMNISTList: A list-like object containing PermutedMNIST tasks.
        """
        permutations = self._generate_task_variations()
        return permuted_mnist.PermutedMNISTList(
            permutations=permutations,
            data_path=datasets_folder,
            use_one_hot=True,
            padding=padding,
            validation_size=validation_size,
        )
