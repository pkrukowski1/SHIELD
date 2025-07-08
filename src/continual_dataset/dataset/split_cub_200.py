"""
Split CUB200 Dataset
^^^^^^^^^^^^^^^^^^^

The module contains a wrapper for data handlers for the SplitCUB200 task.
"""

import torchvision.datasets as datasets
import torch

import os
import time
import urllib.request
import tarfile
import pandas
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import numpy.matlib as npm
from typing import Tuple

from hypnettorch.data.large_img_dataset import LargeImgDataset
from hypnettorch.data.ilsvrc2012_data import ILSVRC2012Data
from hypnettorch.data.dataset import Dataset
from typing import List, Optional, Union

def _transform_split_outputs(data: Tuple[np.ndarray], outputs: np.ndarray) -> np.ndarray:
    """
    Actual implementation of method `transform_outputs` for split dataset
    handlers.

    Args:
        data (Tuple[np.ndarray]): Data handler.
        outputs (np.ndarray): The outputs to be transformed.

    Returns:
        np.ndarray: The transformed outputs.
    """
    if not data._full_out_dim:
        # TODO implement reverse direction as well.
        raise NotImplementedError('This method is currently only ' +
            'implemented if constructor argument "full_out_dim" was set.')

    labels = [label % data._data['num_classes'] for label in data._labels]

    if data.is_one_hot:
        assert(outputs.shape[1] == data._data['num_classes'])
        mask = np.zeros(data._data['num_classes'], dtype=np.bool)
        mask[labels] = True

        return outputs[:, mask]
    else:
        assert (outputs.shape[1] == 1)
        ret = outputs.copy()
        for i, l in enumerate(labels):
            ret[ret == l] = i
        return ret


def get_split_cub200_handlers(
    data_path: str,
    use_one_hot: bool = True,
    validation_size: int = 0,
    num_classes_per_task: int = 40,
    num_tasks: Optional[int] = None,
    trgt_padding: Optional[int] = None
) -> Tuple:
    """
    Instantiate a list of :class:`SplitCUB200Data` objects, each containing a disjoint set of labels.

    The SplitCUB200 task consists of multiple tasks, each corresponding to a subset of images with
    consecutive labels, e.g., {0,...,39}, {40,...,79}, ..., {160,...,199}.

    Args:
        data_path (str): Path to the CUB200 dataset. If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool, optional): Whether the class labels should be represented in a one-hot
            encoding. Defaults to True.
        validation_size (int, optional): The size of the validation set for each individual
            data handler. Defaults to 0.
        num_classes_per_task (int, optional): Number of classes per data handler. Defaults to 40.
        num_tasks (Optional[int], optional): Number of data handlers to return. If None, will be set to
            200 // num_classes_per_task. Defaults to None.
        trgt_padding (Optional[int], optional): If provided, pad the targets with this many fake classes.
            See :class:`SplitCUB200Data` for details. Defaults to None.

    Returns:
        List[SplitCUB200Data]: List of data handlers, each corresponding to a :class:`SplitCUB200Data` object.

    Raises:
        ValueError: If the number of tasks or classes per task is invalid.
    """
    assert num_tasks is None or num_tasks > 0
    if num_tasks is None:
        num_tasks = 200 // num_classes_per_task

    if not (num_tasks >= 1 and (num_tasks * num_classes_per_task) <= 200):
        raise ValueError(
            f'Cannot create SplitCUB200 datasets for {num_tasks} tasks '
            f'with {num_classes_per_task} classes per task.'
        )

    print(f'Creating {num_tasks} data handlers for SplitCUB200 tasks ...')

    handlers: List[SplitCUB200Data] = []
    steps = num_classes_per_task
    for i in range(0, 200, steps):
        handlers.append(SplitCUB200Data(
            data_path,
            use_one_hot=use_one_hot,
            validation_size_per_class=validation_size,
            labels=range(i, i + steps),
            trgt_padding=trgt_padding
        ))

        if len(handlers) == num_tasks:
            break

    print('Creating data handlers for SplitCUB200 tasks ... Done')

    return handlers


class CUB2002011(LargeImgDataset):
    """An instance of the class shall represent the CUB-200-2011 dataset.

    The input data of the dataset will be strings to image files. The output
    data corresponds to object labels (bird categories).

    Note:
        The dataset will be downloaded if not available.

    Note:
        The original category labels range from 1-200. We modify them to
        range from 0 - 199.
    """
    _DOWNLOAD_PATH = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    _IMG_ANNO_FILE = 'CUB_200_2011.tgz'
    _SEGMENTATION_FILE = 'segmentations.tgz' # UNUSED
    # In which subfolder of the datapath should the data be stored.
    _SUBFOLDER = 'cub_200_2011'
    # After extracting the downloaded archive, the data will be in
    # this subfolder.
    _REL_BASE = 'CUB_200_2011'
    _IMG_DIR = 'images' # Realitve to _REL_BASE
    _CLASSES_FILE = 'classes.txt' # Realitve to _REL_BASE
    _IMG_CLASS_LBLS_FILE = 'image_class_labels.txt' # Realitve to _REL_BASE
    _IMG_FILE = 'images.txt' # Realitve to _REL_BASE
    _TRAIN_TEST_SPLIT_FILE = 'train_test_split.txt' # Realitve to _REL_BASE


    def __init__(self,
        data_path,
        use_one_hot=True,
        validation_size_per_class=50,
        seed=1,
        labels=[i for i in range(40)]) -> None:


        super().__init__('')
        start = time.time()

        self._labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dataloder = None

        np.random.seed(seed)

        print('Reading CUB-200-2011 dataset ...')

        # Actual data path
        data_path = os.path.join(data_path, CUB2002011._SUBFOLDER)

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)
            
        full_data_path = os.path.join(data_path, CUB2002011._REL_BASE)
        image_dir = os.path.join(full_data_path, CUB2002011._IMG_DIR)
        classes_fn = os.path.join(full_data_path, CUB2002011._CLASSES_FILE)
        img_class_fn = os.path.join(full_data_path,
                                    CUB2002011._IMG_CLASS_LBLS_FILE)
        image_fn = os.path.join(full_data_path, CUB2002011._IMG_FILE)
        train_test_split_fn = os.path.join(full_data_path,
                                           CUB2002011._TRAIN_TEST_SPLIT_FILE)

        ########################
        ### Download dataset ###
        ########################
        if not os.path.exists(image_dir) or \
                not os.path.exists(classes_fn) or \
                not os.path.exists(img_class_fn) or \
                not os.path.exists(image_fn) or \
                not os.path.exists(train_test_split_fn):
            print('Downloading dataset ...')
            archive_fn = os.path.join(data_path, CUB2002011._IMG_ANNO_FILE)
            urllib.request.urlretrieve(CUB2002011._DOWNLOAD_PATH + \
                                       CUB2002011._IMG_ANNO_FILE, \
                                       archive_fn)
            # Extract downloaded dataset.
            tar = tarfile.open(archive_fn, "r:gz")
            tar.extractall(path=data_path)
            tar.close()

            os.remove(archive_fn)

        ####################
        ### Read dataset ###
        ####################
        # We use the same transforms as 
        train_transform, test_transform = \
            ILSVRC2012Data.torch_input_transforms()

        # Consider all images as training images. We split the dataset later.
        ds_train = datasets.ImageFolder(image_dir, train_transform)

        # Ability to translate image IDs into image paths and back.
        image_ids_csv = pandas.read_csv(image_fn, sep=' ',
                                        names=['img_id', 'img_path'])
        id2img = dict(zip(list(image_ids_csv['img_id']),
                          list(image_ids_csv['img_path'])))
        # Since the ImageFolder class uses absolute paths, we have to change
        # the just read relative paths.
        for iid in id2img.keys():
            id2img[iid] = os.path.join(image_dir, id2img[iid])
        img2id = {v: k for k, v in id2img.items()}

        # Image ID to label.
        img_lbl_csv = pandas.read_csv(img_class_fn, sep=' ',
                                      names=['img_id', 'label'])
        
        id2lbl = dict(zip(list(img_lbl_csv['img_id']),
                          list(img_lbl_csv['label'])))
        # Note, categories go from 1-200. We change them to go from 0 - 199.
        for iid in id2lbl.keys():
            id2lbl[iid] = id2lbl[iid] - 1

        # Image ID to label name.
        img_lbl_name_csv = pandas.read_csv(classes_fn, sep=' ',
                                           names=['label', 'label_name'])
        lbl2lbl_name_tmp = dict(zip(list(img_lbl_name_csv['label']),
                                    list(img_lbl_name_csv['label_name'])))
        # Here, we also have to modify the labels to be within 0-199.
        lbl2lbl_name = {k-1: v for k, v in lbl2lbl_name_tmp.items()}

        # Train-test-split.
        train_test_csv = pandas.read_csv(train_test_split_fn, sep=' ',
                                         names=['img_id', 'is_train'])
        id2train = dict(zip(list(train_test_csv['img_id']),
                            list(train_test_csv['is_train'])))

        self._label_to_name = lbl2lbl_name

        ####################
        ### Sanity check ###
        ####################
        for i, (img_path, lbl) in enumerate(ds_train.samples):
            iid = img2id[img_path]
            assert(id2img[iid] == img_path)
            assert(lbl == id2lbl[iid])

        ################################
        ### Train / val / test split ###
        ################################

        # We take from orig_samples only those labels which are required for
        # the current task
        orig_samples = ds_train.samples
        orig_samples = [sample for sample in orig_samples if sample[1] in labels]
        ds_train.samples = []
        ds_train.imgs = ds_train.samples
        ds_train.targets = []

        ds_test = deepcopy(ds_train)
        ds_test.transform = test_transform
        num_classes = len(labels)

        assert(ds_test.target_transform is None)        

        if validation_size_per_class > 0:
            ds_val = deepcopy(ds_train)
            # NOTE we use test input transforms for the validation set.
            ds_val.transform = test_transform
        else:
            ds_val = None

        val_counts  = np.zeros(num_classes, dtype=np.int)

        for img_path, img_lbl in orig_samples:
            iid = img2id[img_path]
            if id2train[iid] == 1: # In train split.
                if val_counts[img_lbl % num_classes] >= validation_size_per_class: # train sample
                    ds_train.samples.append((img_path, img_lbl))
                else: # validation sample
                    val_counts[img_lbl % num_classes] += 1
                    ds_val.samples.append((img_path, img_lbl))
            else: # In test split.
                ds_test.samples.append((img_path, img_lbl))
        for ds_obj in [ds_train, ds_test] + \
                ([ds_val] if validation_size_per_class > 0 else []):
            ds_obj.targets = [s[1] for s in ds_obj.samples]
            assert(len(ds_obj.samples) == len(ds_obj.imgs) and \
                   len(ds_obj.samples) == len(ds_obj.targets))
                        
        assert(len(ds_train.samples) >= validation_size_per_class*num_classes), \
            "The number of training samples should not be lower than the number of validation samples"
        
        # # Get test labels counter object
        # _test_labels_counter = Counter([sample[1] for sample in ds_test.samples])
        
        
        # while (np.array(list(_test_labels_counter.values())) > 5).any():
        #     for sample in ds_test.samples:
        #         _, y = sample[0], sample[1]
        #         if _test_labels_counter[y] > 5:
        #             ds_train.samples.append(sample)
        #             ds_test.samples.remove(sample)
        #             _test_labels_counter[y] -= 1

        # We use test set as validation set
        ds_val = ds_test

        self._torch_ds_train = ds_train
        self._torch_ds_test = ds_test
        self._torch_ds_val = ds_val


        #####################################
        ### Build internal data structure ###
        #####################################
        num_train = len(self._torch_ds_train.samples)
        num_test = len(self._torch_ds_test.samples)
        num_val = 0 if self._torch_ds_val is None else \
            len(self._torch_ds_val.samples)
        num_samples = num_train + num_test + num_val

        max_path_len = len(max(orig_samples, key=lambda t : len(t[0]))[0])

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = len(labels)
        self._data['is_one_hot'] = use_one_hot

        self._data['in_shape'] = [224, 224, 3]
        self._data['out_shape'] = [len(labels) if use_one_hot else 1]

        self._data['in_data'] = np.chararray([num_samples, 1],
            itemsize=max_path_len, unicode=True)
        for i, (img_path, _) in enumerate(ds_train.samples +
                ([] if num_val == 0 else ds_val.samples) +
                ds_test.samples):
            self._data['in_data'][i, :] = img_path

        labels = np.array(ds_train.targets +
                          ([] if num_val == 0 else ds_val.targets) +
                          ds_test.targets).reshape(-1, 1)

        if use_one_hot:
            labels = self._to_one_hot(labels)
        self._data['out_data'] = labels

        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train + num_val, num_samples)
        if num_val == 0:
            self._data['val_inds'] = None
        else:
            self._data['val_inds'] = np.arange(num_train, num_train + num_val)

        print('Dataset consists of %d training, %d validation and %d test '
              % (num_train, num_val, num_test) + 'samples.')

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))        

        self.train_transform = train_transform
        self.test_transform  = test_transform

    def _to_one_hot(self, labels: Union[np.ndarray, Tuple[int]], reverse: bool=False) -> np.ndarray:
        """
        Transform a list or array of labels into a 1-hot encoding, or decode one-hot labels back to categorical labels.

        Args:
            labels (np.ndarray or list of int): A list or array of class labels.
            reverse (bool): If True, transform one-hot encoded samples back to categorical labels.

        Returns:
            np.ndarray: The one-hot encoded labels, or decoded categorical labels if reverse=True.
        """
        if not self.classification:
            raise RuntimeError('This method can only be called for ' +
                                   'classification datasets.')

        # Initialize encoder.
        if self._one_hot_encoder is None:
            categories = [range(self._labels[0], self._labels[-1]+1)]
            self._one_hot_encoder = OneHotEncoder( \
                categories=categories)
            self._one_hot_encoder.fit(npm.repmat(
                    np.arange(self._labels[0], self._labels[-1]+1), 1, 1).T)

        if reverse:
            # Unfortunately, there is no inverse function in the OneHotEncoder
            # class. Therefore, we take the one-hot-encoded "labels" samples
            # and take the indices of all 1 entries. Note, that these indices
            # are returned as tuples, where the second column contains the
            # original column indices. These column indices from "labels"
            # mudolo the number of classes results in the original labels.
            tmp = np.reshape(np.argwhere(labels)[:,1] % self.num_classes, 
                              (labels.shape[0], -1))
            return tmp + self._labels[0]
        else:
            if self.sequence:
                assert len(self.out_shape) == 1
                num_time_steps = labels.shape[1] # // 1
                n_samples, _ = labels.shape
                labels = labels.reshape(n_samples * num_time_steps, 1)
                labels = self._one_hot_encoder.transform(labels).toarray()
                labels = labels.reshape(n_samples,
                                        num_time_steps * self.num_classes)

                return labels
            else:
                return self._one_hot_encoder.transform(labels).toarray()


    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CUB-200-2011'

    def tf_input_map(self, mode='inference'):
        """Not impemented."""
        # Confirm, whether you wanna process data as in the baseclass or
        # implement a new image loader.
        raise NotImplementedError('Not implemented yet!')

    def _plot_sample(
        self,
        fig: plt.Figure,
        inner_grid,
        num_inner_plots: int,
        ind: int,
        inputs: np.ndarray,
        outputs: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> None:
        """
        Implementation of the abstract method :meth:`data.dataset.Dataset._plot_sample`.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure to plot on.
            inner_grid: The subplot grid specification.
            num_inner_plots (int): Number of inner plots in the grid.
            ind (int): Index of the sample to plot.
            inputs (np.ndarray): Input image(s) to plot.
            outputs (Optional[np.ndarray], optional): Ground truth labels for the sample. Defaults to None.
            predictions (Optional[np.ndarray], optional): Model predictions for the sample. Defaults to None.

        Returns:
            None
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CUB-200-2011 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._label_to_name[label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._label_to_name[pred_label]

                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label) + '\nPrediction: %s (%d)' % \
                             (pred_label_name, pred_label))

        if inputs.size == 1:
            img = self.read_images(inputs)
        else:
            img = inputs

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(img, self.in_shape)))
        fig.add_subplot(ax)

    def input_to_torch_tensor(
        self,
        x: np.ndarray,
        device: Union[str, torch.device],
        mode: str = "inference",
        force_no_preprocessing: bool = False,
        sample_ids: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Convert input numpy arrays to PyTorch tensors, with optional preprocessing.

        This method overrides the base class method. The input data are preprocessed
        (e.g., normalization) if `force_no_preprocessing` is False.

        Args:
            x (np.ndarray): Input data, typically an array of image paths.
            device (Union[str, torch.device]): Device to move the tensor to.
            mode (str, optional): Either "inference" or "train", determines which transform to use.
            force_no_preprocessing (bool, optional): If True, skip preprocessing.
            sample_ids (Optional[np.ndarray], optional): Optional sample indices.

        Returns:
            torch.Tensor: The processed input as a PyTorch tensor.
        """

        if isinstance(x, torch.Tensor):
            return x

        if not force_no_preprocessing:
            if mode == "inference":
                transform = self.test_transform
            elif mode == "train":
                transform = self.train_transform
            else:
                raise ValueError(
                    f"{mode} is not a valid value for the" "argument 'mode'."
                )
            return CUB2002011.torch_preprocess_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(
                self,
                x,
                device,
                mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids,
            )

    @staticmethod
    def torch_preprocess_images(
        x: np.ndarray,
        device: Union[str, torch.device],
        transform,
        img_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> torch.Tensor:
        """
        Preprocess CUB-200 images using a specified PyTorch transformation.

        Args:
            x (np.ndarray): Array of image paths, shape (N, 1).
            device (Union[str, torch.device]): Device to move the resulting tensor to.
            transform: A torchvision transform to apply to each image.
            img_shape (Tuple[int, int, int], optional): Shape of the loaded images. Defaults to (224, 224, 3).

        Returns:
            torch.Tensor: The preprocessed images as a PyTorch tensor.
        """
        assert len(x.shape) == 2
        # First dimension is related to batch size and second is related
        # to the flattened image.
        x = CUB2002011.load_images_to_tensor(x, transform)
        x = x.to(device)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, np.prod(img_shape))
        return x
    
    # Function to load a batch of image paths and convert to tensors
    @staticmethod
    def load_images_to_tensor(image_paths, transform):
        tensors = []
        for path in image_paths:
            # Load the image
            img = Image.open(path[0]).convert('RGB')  # Ensure RGB format
            img_tensor = transform(img)
            tensors.append(img_tensor)

        # Stack tensors into a single batch
        batch_tensor = torch.stack(tensors)
        return batch_tensor
    
    def get_val_inputs(self, dtype="torch"):
        """
        Returns validation inputs.
        """
        assert dtype in ["torch", "numpy"]

        inputs = [sample[0] for sample in self._torch_ds_val.samples]
        inputs = np.array(inputs).reshape(-1,1)

        if dtype == "torch":
            inputs = self.input_to_torch_tensor(
                inputs,
                self.device,
                mode="inference"
            )
        return inputs
    
    def get_val_outputs(self, dtype="torch"):
        """
        Returns validation outputs.
        """
        assert dtype in ["torch", "numpy"]

        outputs = [sample[1] for sample in self._torch_ds_val.samples]
        outputs = self._to_one_hot(np.array(outputs).reshape(-1,1))

        if dtype == "torch":
            outputs = super().output_to_torch_tensor(
                outputs,
                self.device,
                mode="inference"
            )
        return outputs
    
    def get_train_inputs(self, dtype="torch"):
        """
        Returns training inputs.
        """
        assert dtype in ["torch", "numpy"]

        inputs = [sample[0] for sample in self._torch_ds_train.samples]
        inputs = np.array(inputs).reshape(-1,1)

        if dtype == "torch":
            inputs = self.input_to_torch_tensor(
                inputs,
                self.device,
                mode="train"
            )
        return inputs
    
    def get_train_outputs(self, dtype="torch"):
        """
        Returns training outputs.
        """
        assert dtype in ["torch", "numpy"]

        outputs = [sample[1] for sample in self._torch_ds_train.samples]
        outputs = self._to_one_hot(np.array(outputs).reshape(-1,1))

        if dtype == "torch":
            outputs = super().output_to_torch_tensor(
                outputs,
                self.device,
                mode="train"
            )
        return outputs
    
    def get_test_inputs(self, dtype="torch"):
        """
        Returns test inputs.
        """
        assert dtype in ["torch", "numpy"]

        inputs = [sample[0] for sample in self._torch_ds_test.samples]
        inputs = np.array(inputs).reshape(-1,1)

        if dtype == "torch":
            inputs = self.input_to_torch_tensor(
                inputs,
                self.device,
                mode="inference"
            )
        return inputs
    
    def get_test_outputs(self, dtype="torch"):
        """
        Returns test outputs.
        """
        assert dtype in ["torch", "numpy"]
        
        outputs = [sample[1] for sample in self._torch_ds_test.samples]
        outputs = self._to_one_hot(np.array(outputs).reshape(-1,1))

        if dtype == "torch":
            outputs = super().output_to_torch_tensor(
                outputs,
                self.device,
                mode="train"
            )
        return outputs
    
    def output_to_torch_tensor(self, y, device, mode='inference', 
                               force_no_preprocessing=False, sample_ids=None):
        """
        For the description of arguments please see docstring of method :meth:`input_to_torch_tensor`.
        Generally, the function is required to be comaptible with the rest of
        CL dataset handlers.
        """
        return torch.Tensor(y).to(device)
        

class SplitCUB200Data(CUB2002011):
    """
    An instance of the class represents a SplitCUB200 task.

    Args:
        dataset_folder (str): Path to the dataset folder. If the dataset does not exist, it will be downloaded into this folder.
        use_one_hot (bool, optional): Whether the class labels should be represented in a one-hot encoding. Default is False.
        validation_size_per_class (int, optional): Number of validation samples per class. Validation samples will be taken from the training set. Default is 100.
        labels (list or range, optional): The labels that should be part of this task. Default is range(0, 40).
        full_out_dim (bool, optional): If True, use the original CUB200 output dimension instead of the new task output dimension.
            This affects the attributes :attr:`data.dataset.Dataset.num_classes` and :attr:`data.dataset.Dataset.out_shape`.
            Default is False.
        trgt_padding (int, optional): If provided, this many fake classes will be added, so that the returned dataset has
            ``len(labels) + trgt_padding`` classes. All padded classes have no input instances.
            One-hot encodings are padded to fit the new number of classes.
    """
    def __init__(
        self,
        dataset_folder: str,
        use_one_hot: bool = False,
        validation_size_per_class: int = 100,
        labels: Union[List[int], range] = range(0, 40),
        full_out_dim: bool = False,
        trgt_padding: Optional[int] = None
    ) -> None:
        # Note, we build the validation set below!
        super().__init__(dataset_folder, 
                         use_one_hot=use_one_hot, 
                         validation_size_per_class=validation_size_per_class,
                         labels=labels)

        self._full_out_dim = full_out_dim
        if isinstance(labels, range):
            labels = list(labels)
        assert np.all(np.array(labels) >= 0) and \
               len(labels) == len(np.unique(labels))
        K = len(labels)

        self._labels = labels

        train_ins = self.get_train_inputs(dtype="numpy")
        test_ins = self.get_test_inputs(dtype="numpy")

        train_outs = self.get_train_outputs(dtype="numpy")
        test_outs = self.get_test_outputs(dtype="numpy")

        # Get labels.
        if self.is_one_hot:
            train_labels = self._to_one_hot(train_outs, reverse=True)
            test_labels = self._to_one_hot(test_outs, reverse=True)
        else:
            train_labels = train_outs
            test_labels = test_outs

        train_labels = train_labels.squeeze()
        test_labels = test_labels.squeeze()

        train_mask = train_labels == labels[0]
        test_mask = test_labels == labels[0]

        for k in range(1, K):
            train_mask = np.logical_or(train_mask, train_labels == labels[k])
            test_mask = np.logical_or(test_mask, test_labels == labels[k])

        train_ins = train_ins[train_mask, :]
        test_ins = test_ins[test_mask, :]

        train_outs = train_outs[train_mask, :]
        test_outs = test_outs[test_mask, :]

        if validation_size_per_class > 0:
            if validation_size_per_class >= train_outs.shape[0]:
                raise ValueError('Validation set size must be smaller than ' +
                                 '%d.' % train_outs.shape[0])
            val_inds = np.arange(int(validation_size_per_class*len(labels)))
            train_inds = np.arange(int(validation_size_per_class*len(labels)), train_outs.shape[0])

        else:
            train_inds = np.arange(train_outs.shape[0])

        test_inds = np.arange(train_outs.shape[0],
                              train_outs.shape[0] + test_outs.shape[0])

        outputs = np.concatenate([train_outs, test_outs], axis=0)

        if not full_out_dim:
            # Transform outputs, e.g., if 1-hot [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            # Note, the method assumes `full_out_dim` when later called by a
            # user. We just misuse the function to call it inside the
            # constructor.
            self._full_out_dim = True
            outputs = self.transform_outputs(outputs)
            self._full_out_dim = full_out_dim

            # Note, we may also have to adapt the output shape appropriately.
            if self.is_one_hot:
                self._data['out_shape'] = [len(labels)]

        images = np.concatenate([train_ins, test_ins], axis=0)

        ### Overwrite internal data structure. Only keep desired labels.

        # Note, we continue to pretend to be a 10 class problem, such that
        # the user has easy access to the correct labels and has the original
        # 1-hot encodings.
        if not full_out_dim:
            self._data['num_classes'] = len(labels)
        else:
            self._data['num_classes'] = len(labels)
        self._data['in_data'] = images
        self._data['out_data'] = outputs
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size_per_class > 0:
            self._data['val_inds'] = val_inds

        n_val = 0
        if validation_size_per_class > 0:
            n_val = val_inds.size

        if trgt_padding is not None and trgt_padding > 0:
            print('SplitCUB200 targets will be padded with %d zeroes.' \
                  % trgt_padding)
            self._data['num_classes'] += trgt_padding

            if self.is_one_hot:
                self._data['out_shape'] = [self._data['out_shape'][0] + \
                                           trgt_padding]
                out_data = self._data['out_data']
                self._data['out_data'] = np.concatenate((out_data,
                    np.zeros((out_data.shape[0], trgt_padding))), axis=1)

        print('Created SplitCUB200 task with labels %s and %d train, %d test '
              % (str(labels), train_inds.size, test_inds.size) +
              'and %d val samples.' % (n_val))


    def transform_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """
        Transform the outputs from the CUB200 dataset into proper labels
        based on the constructor argument ``labels``.

        The output will have ``len(labels)`` classes.

        Example:
            Split with labels [2, 3]:

            One-hot encodings: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> [0, 1]
            Labels: 3 -> 1

        Args:
            outputs (np.ndarray): 2D numpy array of outputs.

        Returns:
            np.ndarray: 2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)


    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitCUB200'