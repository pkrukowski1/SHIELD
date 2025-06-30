import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch


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
