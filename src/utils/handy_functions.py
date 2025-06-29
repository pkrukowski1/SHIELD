import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def plot_heatmap(load_path):
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
    table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()


def write_pickle_file(filename, object_to_save):
    torch.save(object_to_save.state_dict(), f"{filename}.pt")
