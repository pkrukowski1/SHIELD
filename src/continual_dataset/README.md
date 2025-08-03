## `continual_learning_datasets`

This module provides support for a variety of continual learning benchmarks used in the SHIELD project.

- `dataset/` – Internal logic for managing dataset downloads, preprocessing, task splits, and caching.
- `cl_dataset_abc.py` – Abstract base class defining the interface for all continual learning datasets.
- `cl_permuted_mnist.py` – Implementation of the Permuted MNIST benchmark, where each task applies a fixed pixel permutation.
- `cl_rotated_mnist.py` – Rotated MNIST benchmark, with each task corresponding to a different image rotation angle.
- `cl_split_cifar_100.py` – Split CIFAR-100 dataset, divided into multiple disjoint classification tasks.
- `cl_split_cub_200.py` – Split CUB-200 (Caltech-UCSD Birds) dataset for fine-grained continual classification tasks.
- `cl_split_mini_imagenet.py` – Split MiniImageNet dataset for continual few-shot or standard classification benchmarks.
- `cl_split_mnist.py` – Split MNIST dataset, dividing digits into disjoint tasks (e.g., 0–1, 2–3, etc.).
- `cl_tiny_imagenet.py` – Continual learning benchmark based on the Tiny ImageNet dataset.