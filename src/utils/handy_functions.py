import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.model_abc import CLModuleABC
import torch


def calculate_accuracy(data: np.ndarray, model: CLModuleABC, evaluation_dataset: str, 
                       epsilon: float, number_of_task: int, device: torch.device) -> torch.Tensor:
    
    """
    Calculates the classification accuracy of a neural network model on a specified evaluation dataset.
        model (CLModuleABC): The neural network model to be evaluated. Must have 'hnet' and 'target_network' attributes.
        epsilon (float): Epsilon value for input perturbation.
        number_of_task (int): Task identifier.
    Notes:
        - The function sets the model to evaluation mode.
    """

    model.hnet.eval()
    model.target_network.eval()
  
    if evaluation_dataset == "validation":
        input_data = data.get_val_inputs()
        output_data = data.get_val_outputs()
    elif evaluation_dataset == "test":
        input_data = data.get_test_inputs()
        output_data = data.get_test_outputs()

    test_input = data.input_to_torch_tensor(input_data, device, mode="inference")
    test_output = data.output_to_torch_tensor(output_data, device, mode="inference")

    gt_classes = test_output.max(dim=1)[1]        

    logits, _ = model.target_network.forward(test_input, epsilon, number_of_task)
    predictions = logits.max(dim=1)[1]

    accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0
    return accuracy


def evaluate_previous_tasks(hypernetwork, target_network, dataframe_results, list_of_permutations, parameters):
    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters["number_of_task"] + 1):
        currently_tested_task = list_of_permutations[task]
        hypernetwork_weights = hypernetwork.forward(cond_id=task)

        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork_weights,
            parameters=parameters,
            evaluation_dataset="test",
        )
        result = {
            "after_learning_of_task": parameters["number_of_task"],
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")
        dataframe_results = dataframe_results.append(result, ignore_index=True)
    return dataframe_results


def plot_heatmap(load_path):
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
    table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()
