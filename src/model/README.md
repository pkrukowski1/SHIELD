## `model`

This module defines the neural network architectures used in SHIELD.

- `hypernet_with_alexnet.py` – Implementation of a hypernetwork architecture that generates weights for an AlexNet-based target model.
- `hypernet_with_mlp.py` – Hypernetwork setup for a simple Multi-Layer Perceptron (MLP) target architecture.
- `hypernet_with_resnet18.py` – Defines a hypernetwork that produces parameters for a ResNet-18 target model.
- `hypernet_with_vit.py` – Implements a hypernetwork-based pipeline for generating target weights for a classifier attached at the top of the pretrained ViT model.
- `model_abc.py` – Abstract base class specifying the interface for all hypernetwork-target model combinations.
