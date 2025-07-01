from hypnettorch.data.special.split_cifar import SplitCIFAR100Data

from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class SplitCIFAR100(ContinualLearningTaskGenerator):
    """
    Task generator for Split CIFAR-100.

    This dataset consists of (by default) 10 tasks, each with 10 disjoint classes from CIFAR-100.
    """

    def __init__(
        self,
        no_tasks: int = 10,
        use_augmentation: bool = False,
        use_cutout: bool = False,
        validation_size: int = 500
    ) -> None:
        """
        Initialize the SplitCIFAR100.

        Args:
            no_tasks (int): Number of tasks to split the CIFAR-100 dataset into (default is 10 for a 10x10 split).
            use_augmentation (bool): Whether to use data augmentation during training.
            use_cutout (bool): Whether to apply cutout augmentation (if supported).
            validation_size (int): Number of samples to use for the validation set per task.
        """
        super().__init__()
        self.use_augmentation = use_augmentation
        self.use_cutout = use_cutout
        self.no_classes_per_task = 100 // no_tasks
        self.validation_size = validation_size
        self.no_tasks = no_tasks

    def _generate_task_variations(self) -> None:
        """
        Not used in this generator, as split is handled internally.

        Returns:
            None
        """
        return None

    def prepare_tasks(self, datasets_folder: str):
        """
        Prepare Split CIFAR-100 tasks.

        Args:
            datasets_folder (str): Folder for dataset download/storage.

        Returns:
            List[SplitCIFAR100Data]: List of dataset handlers for each task.
        """
        label_splits = [range(self.no_classes_per_task*i, self.no_classes_per_task*(i+1)) for i in range(self.no_tasks)]
        handlers = [
            SplitCIFAR100Data(
                datasets_folder,
                use_one_hot=True,
                validation_size=self.validation_size,
                use_data_augmentation=self.use_augmentation,
                use_cutout=self.use_cutout,
                labels=label_range,
            )
            for label_range in label_splits
        ]
        return handlers
