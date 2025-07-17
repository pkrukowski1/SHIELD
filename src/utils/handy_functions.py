import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
from copy import deepcopy

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig

from model.model_abc import CLModuleABC
from method.method_abc import MethodABC


def plot_heatmap(load_path: str) -> None:
    """
    Generates and saves a heatmap visualization of accuracy values from a CSV file.

    Reads a CSV file containing columns 'after_learning_of_task', 'tested_task', and 'accuracy',
    pivots the data to create a matrix of accuracy values, and plots a heatmap using seaborn.
    The resulting heatmap is saved as a PDF file with the same name as the input CSV.

    Args:
        load_path (str): Path to the CSV file containing the data. The file should use ';' as a delimiter
                         and have columns 'after_learning_of_task', 'tested_task', and 'accuracy'.

    Returns:
        None
    """
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
    table = dataframe.pivot(index="after_learning_of_task", columns="tested_task", values="accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()


def write_pickle_file(filename: str, object_to_save: torch.nn.Module):
    """
    Saves the state dictionary of a PyTorch object to a file in .pt format.
    Args:
        filename (str): The base name of the file (without extension) to save the state dictionary to.
        object_to_save (torch.nn.Module): The PyTorch object whose state dictionary will be saved.
    Returns:
        None
    Example:
        write_pickle_file("model_checkpoint", model)
        # This will save the model's state dict to 'model_checkpoint.pt'
    """

    torch.save(object_to_save.state_dict(), f"{filename}.pt")

def safe_none(val: Union[None,str]) -> None:
    """
    Converts a string object to None.
    """
    return None if val in ["None", "", None] else val

def make_deepcopy(method: MethodABC, config: DictConfig, device: torch.device) -> CLModuleABC:
    """
    Returns custom deepcopy of the module.
    """
    best_module = instantiate(config.model, number_of_tasks=config.dataset.number_of_tasks)
    best_module.load_state_dict(deepcopy(method.module.state_dict()))
    best_module = best_module.to(device)

    return best_module

def prepare_weights(hnet_weights: dict, model: CLModuleABC) -> dict:
    """
    Prepares weights of a hypernetwork with corrected keys for loading.

    Args:
        hnet_weights (dict): Loaded hypernetwork weights (state dict).
        model (CLModuleABC): Model instance containing hypernetwork.

    Returns:
        dict: Updated state dict compatible with model.hnet.
    """
    return {
        hypernet_key: hnet_weights[loaded_key]
        for (hypernet_key, _), loaded_key in zip(model.hnet.named_parameters(), hnet_weights.keys())
    }


def compute_classical_accuracy_per_task(
    model: CLModuleABC,
    datasets: List,
    fabric,
    config: DictConfig
) -> List[float]:
    """
    Computes classical accuracy for each task in the dataset.

    Classical accuracy is the standard fraction of correct predictions.

    Args:
        model (CLModuleABC): The model to evaluate.
        datasets (List): List of task datasets.
        fabric: Fabric device and setup handler.
        config (DictConfig): Configuration object with experiment params.

    Returns:
        List[float]: Classical accuracy for each task.
    """
    accuracies = []
    for task_id, dataset in enumerate(datasets):
        inputs, targets = dataset.get_test_inputs(), dataset.get_test_outputs()
        test_input = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
        test_target = dataset.output_to_torch_tensor(targets, fabric.device, mode="inference")
        test_target = test_target.max(dim=1)[1]

        with torch.no_grad():
            logits = model(x=test_input, epsilon=config.exp.epsilon, task_id=task_id)[0]
            preds = logits.max(dim=1)[1]
            acc = 100.0 * (preds == test_target).float().mean().item()
            accuracies.append(acc)

    return accuracies
