from continual_dataset.dataset.split_mini_imagenet import SplitMiniImageNet as SplitMiniImageNetData
from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator

from typing import List

class SplitMiniImageNet(ContinualLearningTaskGenerator):
    """
    Task generator for SplitMiniImageNet.

    This version splits the dataset into `number_of_tasks` tasks, each with equal number of unique classes.
    """

    def __init__(
        self,
        number_of_tasks: int,
        use_augmentation: bool = False,
        validation_size: int = 2000,
        train_only_on_first_ten_tasks: bool = True,
        batch_size: int = 16
    ) -> None:
        """
        Initialize the SplitMiniImageNet generator.

        Args:
            number_of_tasks (int): Number of tasks to split the dataset into.
            use_augmentation (bool, optional): Whether to use data augmentation. Defaults to False.
            validation_size (int, optional): Number of validation samples per task. Defaults to 2000.
            train_only_on_first_ten_tasks (bool, optional): If true, a training is interrupted after
                learning first ten tasks as in https://arxiv.org/pdf/2402.11196
            batch_size (int, optional): Batch size.

        Attributes:
            number_of_tasks (int): Number of tasks to split the dataset into.
            use_augmentation (bool): Whether to use data augmentation.
            validation_size (int): Number of validation samples per task.
            train_only_on_first_ten_tasks (bool, optional): If true, a training is interrupted after
                learning first ten tasks as in https://arxiv.org/pdf/2402.11196
            batch_size (int, optional): Batch size.
        """
        super().__init__()

        self.number_of_tasks = number_of_tasks
        self.use_augmentation = use_augmentation
        self.validation_size = validation_size
        self.train_only_on_first_ten_tasks = train_only_on_first_ten_tasks

        self.batch_size = batch_size
        self.no_classes_per_task = 100 // self.number_of_tasks

    def _generate_task_variations(self) -> None:
        """
        Not used in this generator, as split is handled internally.

        Returns:
            None
        """
        
        return None

    def prepare_tasks(self, datasets_folder: str) -> List:
        """
        Prepare Split-miniImageNet tasks.

        Args:
            datasets_folder (str): Path to the dataset storage directory.

        Returns:
            List[continual_dataset.dataset.split_mini_imagenet.SplitMiniImageNet]: A list of dataset handlers for each task.
        """
        handlers = []
        for i in range(self.number_of_tasks):
            if self.train_only_on_first_ten_tasks and i == 10:
                break
            no_validation_samples_per_class = self.validation_size // self.no_classes_per_task
            handlers.append(
                SplitMiniImageNetData(
                    path=datasets_folder,
                    no_classes_per_task=self.no_classes_per_task,
                    use_one_hot=True,
                    use_data_augmentation=self.use_augmentation,
                    validation_size=no_validation_samples_per_class,
                    task_id=i,
                    batch_size=self.batch_size
                )
            )

        return handlers
