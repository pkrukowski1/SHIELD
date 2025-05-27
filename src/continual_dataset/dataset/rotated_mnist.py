"""
Implementation of RotatedMNIST for continual learning tasks.
"""

import copy
import numpy as np
from hypnettorch.data.mnist_data import MNISTData

class RotatedMNISTlist():
    def __init__(self, angles, data_path, use_one_hot=True,
                 validation_size=0, padding=0, trgt_padding=None,
                 show_angle_change_msg=True):
        print('Loading MNIST into memory, shared among %d rotation tasks.' %
              len(angles))

        self._data = RotatedMNIST(data_path, use_one_hot=use_one_hot,
            validation_size=validation_size, angle=None, padding=padding,
            trgt_padding=trgt_padding)

        self._angles = angles
        self._show_angle_change_msg = show_angle_change_msg

        self._batch_gens_train = [None] * len(angles)
        self._batch_gens_test = [None] * len(angles)
        self._batch_gens_val = [None] * len(angles)

        assert hasattr(self._data, '_batch_gen_train') and \
            self._data._batch_gen_train is None
        assert hasattr(self._data, '_batch_gen_test') and \
            self._data._batch_gen_test is None
        assert hasattr(self._data, '_batch_gen_val') and \
            self._data._batch_gen_val is None

        self._active_angle = -1

    def __len__(self):
        return len(self._angles)

    def __getitem__(self, index):
        color_start = '\033[93m'
        color_end = '\033[0m'
        help_msg = 'To disable this message, set "show_angle_change_msg=False".'

        if isinstance(index, slice):
            new_list = copy.copy(self)
            new_list._angles = self._angles[index]
            new_list._batch_gens_train = self._batch_gens_train[index]
            new_list._batch_gens_test = self._batch_gens_test[index]
            new_list._batch_gens_val = self._batch_gens_val[index]

            if self._show_angle_change_msg:
                indices = list(range(*index.indices(len(self))))
                print(color_start + 'RotatedMNISTList: A slice with angles ' +
                      str(indices) + ' has been created.' + color_end + help_msg)

            return new_list

        assert isinstance(index, int)

        if self._active_angle != -1:
            self._batch_gens_train[self._active_angle] = \
                self._data._batch_gen_train
            self._batch_gens_test[self._active_angle] = \
                self._data._batch_gen_test
            self._batch_gens_val[self._active_angle] = self._data._batch_gen_val

        self._data.angle = self._angles[index]
        self._data._batch_gen_train = self._batch_gens_train[index]
        self._data._batch_gen_test = self._batch_gens_test[index]
        self._data._batch_gen_val = self._batch_gens_val[index]
        self._active_angle = index

        if self._show_angle_change_msg:
            print(color_start + 'RotatedMNISTList: Data rotated to %d°.' %
                  self._angles[index] + color_end + help_msg)

        return self._data

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

class RotatedMNIST(MNISTData):
    def __init__(self, data_path, use_one_hot=True, validation_size=0,
                 angle=None, padding=0, trgt_padding=None):
        super().__init__(data_path, use_one_hot=use_one_hot,
                         validation_size=validation_size,
                         use_torch_augmentation=False)

        self._padding = padding
        self._input_dim = (28 + padding * 2) ** 2
        self._angle = angle

        if trgt_padding is not None and trgt_padding > 0:
            print('RotatedMNIST targets will be padded with %d zeroes.' % trgt_padding)
            self._data['num_classes'] += trgt_padding

            if self.is_one_hot:
                self._data['out_shape'] = [self._data['out_shape'][0] + trgt_padding]
                out_data = self._data['out_data']
                self._data['out_data'] = np.concatenate((out_data,
                    np.zeros((out_data.shape[0], trgt_padding))), axis=1)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self._transform = RotatedMNIST.torch_input_transforms(
            padding=self._padding, angle=value)

    @property
    def torch_in_shape(self):
        return [self.in_shape[0] + 2 * self._padding,
                self.in_shape[1] + 2 * self._padding, self.in_shape[2]]

    def get_identifier(self):
        return 'RotatedMNIST'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        if not force_no_preprocessing:
            from torch import stack
            img_size = 28 + 2 * self._padding
            x = (x * 255.0).astype('uint8')
            x = x.reshape(-1, 28, 28, 1)

            x = stack([self._transform(x[i, ...]) for i in range(x.shape[0])]).to(device)
            x = x.permute(0, 2, 3, 1)
            x = x.contiguous().view(-1, img_size ** 2)
            return x
        else:
            return MNISTData.input_to_torch_tensor(self, x, device, mode=mode,
                force_no_preprocessing=force_no_preprocessing, sample_ids=sample_ids)

    @staticmethod
    def torch_input_transforms(angle=None, padding=0):
        import torchvision.transforms as transforms

        def _rotate_image(image, angle):
            if angle is None:
                return image
            return transforms.functional.rotate(image, angle)

        transform = transforms.Compose([
            transforms.ToPILImage('L'),
            transforms.Pad(padding),
            transforms.Lambda(lambda img: _rotate_image(img, angle)),
            transforms.ToTensor()
        ])

        return transform

    def tf_input_map(self, mode='inference'):
        raise NotImplementedError('No TensorFlow support for this class.')

if __name__ == '__main__':
    pass
