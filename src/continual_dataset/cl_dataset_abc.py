from abc import ABC, abstractmethod
from typing import Any


class ContinualLearningTaskGenerator(ABC):
    """
    Abstract base class for continual learning dataset generators.

    Subclasses must implement the methods to generate transformations
    (e.g., permutations, rotations) and to prepare the dataset tasks.
    """

    def __init__(self):
        """
        Initialize the generator.
        """

        super().__init__()

    @abstractmethod
    def _generate_task_variations(self) -> list:
        """
        Generate task-specific variations (e.g., permutations, rotations).

        Returns:
            list: A list of transformation specifications.
        """
        pass

    @abstractmethod
    def prepare_tasks(self, datasets_folder: str) -> Any:
        """
        Prepare and return the list of tasks.

        Args:
            datasets_folder (str): Folder path for dataset storage or download.

        Returns:
            Any: A list-like object containing task datasets.
        """
        pass
