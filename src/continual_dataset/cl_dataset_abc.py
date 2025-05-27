from abc import ABC, abstractmethod
from typing import Any


class ContinualLearningTaskGenerator(ABC):
    """
    Abstract base class for continual learning dataset generators.

    Subclasses must implement the methods to generate transformations
    (e.g., permutations, rotations) and to prepare the dataset tasks.
    """

    def __init__(self, number_of_tasks: int, seed: int):
        """
        Initialize the generator.

        Args:
            number_of_tasks (int): Number of distinct tasks to generate.
            seed (int, optional): Random seed for reproducibility.
        """
        self.number_of_tasks = number_of_tasks
        self.seed = seed

    @abstractmethod
    def _generate_task_variations(self) -> list:
        """
        Generate task-specific variations (e.g., permutations, rotations).

        Returns:
            list: A list of transformation specifications.
        """
        pass

    @abstractmethod
    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 0) -> Any:
        """
        Prepare and return the list of tasks.

        Args:
            datasets_folder (str): Folder path for dataset storage or download.
            padding (int): Optional padding for input images.
            validation_size (int): Optional number of validation samples per task.

        Returns:
            Any: A list-like object containing task datasets.
        """
        pass
