from continual_dataset.dataset.split_cub_200 import SplitCUB200Data
from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator

from typing import List

class SplitCUB200(ContinualLearningTaskGenerator):
    """
    Task generator for SplitCUB200.

    This version splits the dataset into `number_of_tasks` tasks, each with equal number of unique classes.
    """

    def __init__(
        self,
        number_of_tasks: int,
        validation_size: int = 0,
    ) -> None:
        
        super().__init__()

        self.number_of_tasks = number_of_tasks
        self.validation_size = validation_size

        self.no_classes_per_task = 200 // self.number_of_tasks

    def _generate_task_variations(self) -> None:
        """
        Not used in this generator, as split is handled internally.

        Returns:
            None
        """
        
        return None

    def prepare_tasks(self, datasets_folder: str) -> List:
        """
        Prepare Split CUB200 tasks.

        Args:
            datasets_folder (str): Path to the dataset storage directory.

        Returns:
            List[continual_dataset.dataset.split_cub_200.SplitCUB200]: A list of dataset handlers for each task.
        """
        handlers = []
        no_classes = 200 // self.number_of_tasks

        for i in range(0, 200, no_classes):
            no_validation_samples_per_class = self.validation_size // self.no_classes_per_task
            handlers.append(SplitCUB200Data(
                datasets_folder,
                use_one_hot=True,
                validation_size_per_class=no_validation_samples_per_class,
                labels=range(i, i + no_classes)
            ))
        return handlers
