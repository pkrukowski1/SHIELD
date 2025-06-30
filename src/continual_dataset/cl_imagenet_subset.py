from continual_dataset.dataset.imagenet_subset import SubsetImageNet
from continual_dataset.cl_dataset_abc import ContinualLearningTaskGenerator


class ImageNetSubset(ContinualLearningTaskGenerator):
    """
    Task generator for ImageNet-Subset.

    This version splits the dataset into `number_of_tasks` tasks, each with equal number of unique classes.
    """

    def __init__(
        self,
        seed: int = 42,
        use_augmentation: bool = False,
        input_shape: int = 224,
        number_of_tasks: int = 5
    ):
        """
        Initialize the ImageNetSubsetTaskGenerator.

        Args:
            seed (int): Seed for reproducibility (used only if relevant later).
            use_augmentation (bool): Whether to use data augmentation.
            input_shape (int): Expected input shape for the dataset (default 224).
            number_of_tasks (int): Number of tasks (default 5).
        """
        super().__init__(number_of_tasks=number_of_tasks, seed=seed)
        self.use_augmentation = use_augmentation
        self.input_shape = input_shape

    def _generate_task_variations(self):
        """
        Not applicable for this dataset—returns task indices instead.

        Returns:
            List[int]: A list of task indices (e.g., [0, 1, 2, 3, 4]).
        """
        return list(range(self.number_of_tasks))

    def prepare_tasks(self, datasets_folder: str, padding: int = 0, validation_size: int = 0):
        """
        Prepare ImageNet-Subset tasks based on the predefined split (FeCAM setup).

        Args:
            datasets_folder (str): Path to the dataset storage directory.
            padding (int): Unused for ImageNet-Subset, kept for interface consistency.
            validation_size (int): Number of validation samples per task.

        Returns:
            List[SubsetImageNet]: A list of dataset handlers for each task.
        """
        task_indices = self._generate_task_variations()
        handlers = []

        for task_no in task_indices:
            handlers.append(
                SubsetImageNet(
                    data_path=datasets_folder,
                    number_of_task=task_no,
                    validation_size=validation_size,
                    use_data_augmentation=self.use_augmentation,
                    use_one_hot=True,
                    input_shape=self.input_shape,
                )
            )

        return handlers
