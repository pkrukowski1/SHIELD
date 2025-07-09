from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning.fabric import Fabric

import torch


def setup_fabric(config: DictConfig) -> Fabric:
    """
    Sets up Fabric run based on config.
    """

    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fabric.launch()
    return fabric