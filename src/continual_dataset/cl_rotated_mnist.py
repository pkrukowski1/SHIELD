import numpy as np

from continual_dataset.cl_rotated_mnist import RotatedMNIST
from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class RotatedMNISTTask(ContinualLearningTaskGenerator):
    """
    A utility class to generate a list of RotatedMNIST tasks for continual learning.

    Each task corresponds to a unique random rotation applied to the MNIST images,
    helping simulate different tasks in a continual learning setup.
    """

    def __init__(self, number_of_tasks: int, seed: int = 42):
        """
        Initialize the RotatedMNISTTaskGenerator.

        Args:
            number_of_tasks (int): The total number of rotated tasks to generate.
            seed (int): Random seed for reproducibility (default 42).
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)

    def _generate_task_variations(self):
        """
        Generate random rotations (in degrees) for each task.

        Returns:
            List[float]: A list of rotation angles in degrees.
        """
        return [np.random.uniform(0, 360) for _ in range(self.number_of_tasks)]

    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 0):
        """
        Create a list of RotatedMNIST tasks using the generated rotation angles.

        Args:
            datasets_folder (str): Path to store or load the MNIST dataset.
            padding (int, optional): Padding to apply to each image. Defaults to 0.
            validation_size (int, optional): Number of validation samples per task. Defaults to 0.

        Returns:
            RotatedMNIST: A list-like object containing RotatedMNIST tasks.
        """
        rotations = self._generate_task_variations()
        return RotatedMNIST(
            rotations=rotations,
            data_path=datasets_folder,
            use_one_hot=True,
            padding=padding,
            validation_size=validation_size,
        )
