"""
Implementation of ImageNet-Subset for continual learning tasks.

The order of classes, augmentation policy and continual learning task setup
were set as in FeCAM: https://github.com/dipamgoswami/FeCAM.
"""

from typing import Union
import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder
from hypnettorch.data.dataset import Dataset


class LoadImage:
    def __call__(self, path):
        return Image.open(path).convert("RGB")


class SubsetImageNet(Dataset):
    def __init__(
        self,
        data_path: str,
        number_of_task: int,
        use_one_hot: bool = False,
        use_data_augmentation: bool = False,
        validation_size: int = 100,
        seed: int = 1,
        input_shape: int = 64,
        class_order: Union[tuple, None] = None,
    ):
        """
        Initializes the class with the ImageNet-Subset dataset.

        Parameters:
        -----------
            *data_path* (str) path to the directory containing
                        'seed_1993_subset_100_imagenet/data/' with 'train'
                        and 'val' subfolders
            *number_of_task* (int) describes the currently considered task
            *use_one_hot* (bool, Optional) describes whether one-hot encoding
                          on the output classes should be performed
            *use_data_augmentation* (bool, Optional) describes whether data
                                    augmentation methods should be applied,
                                    according to the FeCAM techniques
            *validation_size* (int) defines the total size of the validation
                              set, at least 20 elements (1 per each class);
                              0 if the validation set should not be used
            *seed* (int) defines the seed value
            *input_shape* (int) size of the image (width and height)
            *class_order* (list / None) if a tuple is given, the order
                          of classes is changed; if None, a default order
                          from FeCAM is considered
        """
        # FIXME: How many elements may be in the validation set?
        assert validation_size >= 20 or validation_size == 0
        assert number_of_task in [0, 1, 2, 3, 4]

        super().__init__()
        print("Reading ImageNet-Subset")
        self._data_path = f"{data_path}/seed_1993_subset_100_imagenet/data/"
        self._train_path = f"{self._data_path}train/"
        self._test_path = f"{self._data_path}val/"
        if class_order is None:
            self._class_order = (
                68, 56, 78,  8, 23, 84, 90, 65, 74, 76,
                40, 89,  3, 92, 55,  9, 26, 80, 43, 38,
                58, 70, 77,  1, 85, 19, 17, 50, 28, 53,
                13, 81, 45, 82,  6, 59, 83, 16, 15, 44,
                91, 41, 72, 60, 79, 52, 20, 10, 31, 54,
                37, 95, 14, 71, 96, 98, 97,  2, 64, 66,
                42, 22, 35, 86, 24, 34, 87, 21, 99,  0,
                88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
                62, 69, 36, 61,  7, 63, 75,  5, 32,  4,
                51, 48, 73, 93, 39, 67, 29, 49, 57, 33
            )
        else:
            self._class_order = class_order
        self._number_of_task = number_of_task
        self._use_one_hot = use_one_hot
        self._validation_size = validation_size
        self._use_data_augmentation = use_data_augmentation
        self._seed = seed
        self._input_shape = input_shape
        self._data = dict()
        self._data["imagenet-subset"] = dict()
        self._data["in_shape"] = [self._input_shape, self._input_shape, 3]
        self._data["classification"] = True
        self._data["sequence"] = False
        self._data["num_classes"] = 20
        self._data["is_one_hot"] = self._use_one_hot
        self._data["out_shape"] = [20 if self._use_one_hot else 1]

        self._test_transform = [
            LoadImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize((self._input_shape, self._input_shape)),
        ]
        self._general_transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
           # transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0))),
        ]
        if not self._use_data_augmentation:
            self._train_transform = self._test_transform
        else:
            self._train_transform = [
                LoadImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self._input_shape, self._input_shape)),
            ]
        self._train_transform = transforms.Compose(
            [*self._train_transform, *self._general_transform]
        )
        self._test_transform = transforms.Compose(
            [*self._test_transform, *self._general_transform]
        )

        self.train_data = ImageFolder(self._train_path)
        self.test_data = ImageFolder(self._test_path)
        self.train_data, self.train_labels = self._extract_images_and_labels(
            self.train_data)
        self.test_data, self.test_labels = self._extract_images_and_labels(
            self.test_data)
        self.train_data, self.train_labels, self.test_data, self.test_labels = \
            self._extract_data_for_task(self._number_of_task)
        # Changes real labels to temporary labels to get labels ordered from 0
        # to the number of classes being present in a given task
        self._translate_labels(self.train_labels)
        self.train_labels = self._convert_labels_with_dictionary(
            self.train_labels)
        self.test_labels = self._convert_labels_with_dictionary(
            self.test_labels)
        if self._use_one_hot:
            self.train_labels = self._to_one_hot(
                self.train_labels, reverse=False
            )
            self.test_labels = self._to_one_hot(
                self.test_labels, reverse=False)
        # Prepare training, validation and test sets in the form accepted
        # by hypnettorch
        self._prepare_train_val_test_set()

    def _extract_images_and_labels(self, dataset: ImageFolder):
        X, y = [], []
        for image, target in dataset.imgs:
            X.append(image)
            y.append(target)
        X, y = np.array(X), np.array(y)
        return X, y

    def _extract_data_for_task(self, no_of_task: int):
        """
        Select indices specific for classes from a given task
        and choose the corresponding samples
        """
        current_labels = self._class_order[
            no_of_task * 20:(no_of_task + 1) * 20]
        train_indices = np.argwhere(
            np.isin(self.train_labels, current_labels)
        )
        X_train = self.train_data[train_indices]
        y_train = self.train_labels[train_indices]
        test_indices = np.argwhere(
            np.isin(self.test_labels, current_labels)
        )
        X_test = self.test_data[test_indices]
        y_test = self.test_labels[test_indices]
        return X_train, y_train, X_test, y_test

    def _translate_labels(self, labels):
        sorted_labels = np.unique(np.sort(labels))
        self.translate_temp_label_to_real_labels = dict()
        self.translate_real_label_to_temp_label = dict()
        for i in range(sorted_labels.shape[0]):
            self.translate_temp_label_to_real_labels[i] = sorted_labels[i]
            self.translate_real_label_to_temp_label[sorted_labels[i]] = i

    def translate_temporary_to_real_label(self):
        return self.translate_temp_label_to_real_labels

    def translate_real_to_temporary_label(self):
        return self.translate_real_label_to_temp_label

    def _convert_labels_with_dictionary(self, labels):
        temp_labels = labels.flatten()
        for i in range(temp_labels.shape[0]):
            temp_labels[i] = self.translate_real_label_to_temp_label[
                temp_labels[i]]
        labels = np.expand_dims(temp_labels, axis=1)
        return labels

    def _prepare_train_val_test_set(self):
        """
        Prepares a stratified selection of the training and validation set.
        Also, prepares a final version of the test set and filles keys
        necessary for further calculations.
        """
        if self._validation_size > 0:
            no_of_classes = self._data["num_classes"]
            # We assume that validation samples will be equally distributed
            # among all classes
            self._no_of_val_samples_per_class = self._validation_size // no_of_classes
            (
                self._data["train_inds"],
                self._data["val_inds"],
            ) = self._select_val_indices()

        else:
            self.train_labels = self.train_labels.squeeze()
            self._data["train_inds"] = np.arange(0, self.train_labels.shape[0])

        self._data["test_inds"] = np.arange(
            self.train_labels.shape[0],
            self.train_labels.shape[0] + self.test_labels.shape[0],
        )

        self._data["in_data"] = np.concatenate(
            [self.train_data, self.test_data]
        )
        del self.train_data
        del self.test_data

        if not self._use_one_hot:
            self.train_labels = np.expand_dims(self.train_labels, axis=1)
        self._data["out_data"] = np.concatenate(
            [self.train_labels, self.test_labels]
        )
        del self.train_labels
        del self.test_labels

    def _select_val_indices(self):
        """
        Prepare a selection of train and validation sets with memory saving!

        Returns:
        --------
          *train_indices*: (list) contains indices of elements in the training
                           set
          *test_indices*: (list) contains indices of elements in the test set
        """
        if not self._use_one_hot:
            self.train_labels = self.train_labels.squeeze()
            train_labels_for_val_separation = self.train_labels
        else:
            train_labels_for_val_separation = self._to_one_hot(
                self.train_labels, reverse=True
            ).squeeze()
        unique_classes = np.unique(train_labels_for_val_separation)
        class_positions = {}
        for no_of_class in unique_classes:
            class_positions[no_of_class] = np.argwhere(
                train_labels_for_val_separation == no_of_class
            )

        np.random.seed(self._seed)
        train_indices, val_indices = [], []
        for cur_class in list(class_positions.keys()):
            perm = np.random.permutation(class_positions[cur_class])
            cur_class_val_indices = perm[: self._no_of_val_samples_per_class]
            cur_class_train_indices = perm[self._no_of_val_samples_per_class:]
            train_indices.extend(list(cur_class_train_indices.flatten()))
            val_indices.extend(list(cur_class_val_indices.flatten()))
        return np.array(train_indices), np.array(val_indices)

    def input_to_torch_tensor(
        self,
        x,
        device,
        mode="inference",
        force_no_preprocessing=False,
        sample_ids=None,
    ):
        """
        Prepare mapping of Numpy arrays to PyTorch tensors.
        This method overwrites the method from the base class.
        The input data are preprocessed (data standarization).

        Arguments:
        ----------
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if not force_no_preprocessing:
            if mode == "inference":
                transform = self._test_transform
            elif mode == "train":
                transform = self._train_transform
            else:
                raise ValueError(
                    f"{mode} is not a valid value for the" "argument 'mode'."
                )
            return SubsetImageNet.torch_preprocess_images(
                x, device, transform, img_shape=self.in_shape)
        return Dataset.input_to_torch_tensor(
            self,
            x,
            device,
            mode=mode,
            force_no_preprocessing=force_no_preprocessing,
            sample_ids=sample_ids,
        )

    @staticmethod
    def torch_preprocess_images(x, device, transform, img_shape=[64, 64, 3]):
        """
        Prepare preprocessing of ImageNet-Subset images with a selected
        PyTorch transformation.

        Arguments:
        ----------
            x (Numpy array): 2D Numpy array containing paths to files with images
                             (shape: batch size, 1)
            device (torch.device or int): PyTorch device on which a final
                                          tensor will be moved
            transform: (torchvision.transforms): a method of data modification

        Returns:
        --------
            (torch.Tensor): The preprocessed images as PyTorch tensor.
        """
        features = torch.stack(
            [transform(x[i][0]) for i in range(x.shape[0])]).to(device)
        features = features.permute(0, 2, 3, 1)
        features = features.contiguous().view(-1, np.prod(img_shape))
        return features

    def _plot_sample(self):
        pass

    def get_identifier(self):
        return "ImageNetSubset"

    def _validity_control(self):
        """
        Control whether the set was prepared according to the desired hyperparams.
        """
        # Test set: 20 classes in each task, 50 samples per class
        assert self._data["test_inds"].shape[0] == 1000
        assert self._data["in_data"].shape[0] == \
            self._data["out_data"].squeeze().shape[0]
        if not self._use_one_hot:
            labels_squeezed = self._data["out_data"].squeeze()
        else:
            labels_squeezed = self._to_one_hot(
                self._data["out_data"], reverse=True
            ).squeeze()
        # Control test set
        test_labels = labels_squeezed[self._data["test_inds"]]
        temporary_labels = list(self.translate_temp_label_to_real_labels.keys())
        for label in temporary_labels:
            assert np.count_nonzero(test_labels == label) == 50
        # Control validation set
        if self._validation_size > 0:
            assert self._data["val_inds"].shape[0] == self._no_of_val_samples
            val_labels = labels_squeezed[self._data["val_inds"]]
            for label in temporary_labels:
                assert (
                    np.count_nonzero(val_labels == label)
                    == self._no_of_val_samples_per_class
                )
        # Different number of training samples may be present in various classes


if __name__ == "__main__":
    pass
