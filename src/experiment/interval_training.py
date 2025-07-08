from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Optional, List
import os
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
import logging

import torch

from utils.fabric import setup_fabric
from utils.handy_functions import write_pickle_file, plot_heatmap
from method.method_abc import MethodABC
from model.model_abc import CLModuleABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def experiment(config: DictConfig) -> None:
    """
    Full training and testing on given scenario.
    Initializes datasets, fabric, model, method, and handles training and evaluation
    across tasks. Logs metrics and saves outputs via wandb and to local files.
    """

    number_of_tasks = config.dataset.number_of_tasks

    log.info(f'Initializing scenarios')
    cl_dataset = instantiate(config.dataset)
    task_datasets = cl_dataset.prepare_tasks(os.getenv("DATA_DIR"))

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.model, number_of_tasks=number_of_tasks))

    log.info(f'Setting up method')
    method = instantiate(config.method, module=model)

    dataframe = pd.DataFrame(columns=["after_learning_of_task", "tested_task", "accuracy"])

    all_accuracies = []
    
    try:
        for task_id in range(number_of_tasks):

            method.setup_task(task_id)

            best_hnet, best_target_network = train_single_task(
                method=method,
                task_id=task_id,
                task_datasets=task_datasets,
                config=config,
                device=fabric.device
            )

            dataframe = evaluate_previous_tasks(
                hnet=best_hnet,
                target_network=best_target_network,
                dataframe=dataframe,
                task_id=task_id,
                task_datasets=task_datasets,
                epsilon=method.base_epsilon,
                device=fabric.device
            )

            all_accuracies.append(
                dataframe[dataframe["after_learning_of_task"] == task_id]["accuracy"].tolist()
            )

            dataframe = dataframe.astype({"after_learning_of_task": "int", "tested_task": "int"})
            dataframe.to_csv(f'{config.exp.log_dir}/results.csv', sep=";")

            if wandb.run:
                df_task = dataframe[dataframe["after_learning_of_task"] == task_id]
                wandb.log({
                    f"acc_after_task_{task_id}": wandb.Table(columns=df_task.columns.tolist(), data=df_task.values.tolist())
                })

            # Log average accuracy after learning up to task_id
            accs_up_to_now = dataframe[dataframe["after_learning_of_task"] == task_id]["accuracy"]
            avg_acc = accs_up_to_now.mean()
            log.info(f"Average accuracy after task {task_id}: {avg_acc:.2f}%")
            if wandb.run:
                wandb.log({f"avg_accuracy_up_to_task_{task_id}": avg_acc})
    except IndexError:
        pass


    write_pickle_file(f'{config.exp.log_dir}/hnet', method.module.hnet.weights)

    plot_heatmap(f'{config.exp.log_dir}/results.csv')

    bwt = calculate_backward_transfer(dataframe)
    log.info(f"Backward transfer: {bwt:.5f}")
    if wandb.run:
        wandb.log({"backward_transfer": bwt})

    plot_accuracy_progression(all_accuracies, f"{config.exp.log_dir}/accuracy_progression.png")


def should_log(iteration: int, total_no_iterations: int, no_epochs: Optional[int], no_iterations_per_epoch: Optional[int]) -> bool:
    """
    Determine if logging or validation should be performed at this iteration.
    """
    return (
        (iteration % 100 == 0)
        or (iteration == total_no_iterations - 1)
        or (no_epochs is not None and ((iteration + 1) % no_iterations_per_epoch == 0))
    )

def maybe_log_epoch(iteration: int, no_epochs: Optional[int], no_iterations_per_epoch: Optional[int]) -> None:
    """
    Optionally log the current epoch number.
    """
    if no_epochs is not None and no_iterations_per_epoch:
        current_epoch = (iteration + 1) // no_iterations_per_epoch
        log.info(f"Current epoch: {current_epoch}")

def log_metrics(iteration: int, task_id: int, loss: torch.Tensor, accuracy: float, 
                no_incorrect_hypercubes: int, epsilon: float) -> None:
    """
    Log current training loss and validation metrics.
    """
    log.info(
        f"Task {task_id}, "
        f"iteration: {iteration + 1}, "
        f"train loss: {loss.item():.5f}, "
        f"validation accuracy: {accuracy:.5f}, "
        f"no incorrectly classified hypercubes: {no_incorrect_hypercubes}, "
        f"epsilon: {epsilon:.8f}"
    )
    if wandb.run:
        wandb.log({
            f"train/loss_task_{task_id}": loss.item(),
            f"val/accuracy_task_{task_id}": accuracy,
            f"val/incorrect_hypercubes_task_{task_id}": no_incorrect_hypercubes,
        })

def should_update_best(accuracy: float, best_val_accuracy: float, iteration: int, total_no_iterations: int) -> bool:
    """
    Check if the current model is the best so far.
    """
    return accuracy > best_val_accuracy and iteration >= total_no_iterations // 2

def maybe_step_scheduler(method: MethodABC, iteration: int, total_no_iterations: int, accuracy: float, no_epochs: Optional[int]) -> None:
    """
    Step the learning rate scheduler if applicable.
    """
    if (
        no_epochs is not None
        and method.scheduler is not None
        and ((iteration + 1) % total_no_iterations == 0)
    ):
        log.info("Finishing the current epoch")
        method.make_scheduler_step(accuracy)

def train_single_task(method: MethodABC, task_id: int, task_datasets: Iterable, config: DictConfig, device: torch.device) -> CLModuleABC:
    """
    Train the model for a single task.
    """
    no_iterations = config.exp.no_iterations
    no_epochs = config.exp.no_epochs
    batch_size = config.exp.batch_size

    assert no_iterations is not None or no_epochs is not None, "Arguments `no_iterations` and `no_epochs` cannot both be `None`"

    current_dataset_instance = task_datasets[task_id]

    if no_epochs is not None:
        no_iterations_per_epoch, total_no_iterations = calculate_number_of_iterations(
            number_of_samples=current_dataset_instance.num_train_samples,
            batch_size=batch_size,
            number_of_epochs=no_epochs
        )
    else:
        no_iterations_per_epoch = None
        total_no_iterations = no_iterations

    method.set_no_iterations(total_no_iterations)

    best_hnet = deepcopy(method.module.hnet)
    best_target_network = deepcopy(method.module.target_network)
    best_val_accuracy = 0.0

    log.info(f"Train the {task_id}-th task")

    for iteration in range(total_no_iterations):
        method.module.hnet.train()

        current_batch = current_dataset_instance.next_train_batch(batch_size)

        tensor_input = current_dataset_instance.input_to_torch_tensor(current_batch[0], device, mode="train")
        tensor_output = current_dataset_instance.output_to_torch_tensor(current_batch[1], device, mode="train")
        gt_output = tensor_output.max(dim=1)[1]

        loss, worst_case_prediction = method.forward(tensor_input, gt_output, task_id)
        loss = loss.mean()
        method.backward(loss)
        
        if should_log(iteration, total_no_iterations, no_epochs, no_iterations_per_epoch):
            maybe_log_epoch(iteration, no_epochs, no_iterations_per_epoch)

            accuracy = calculate_accuracy(
                current_dataset_instance,
                hnet=method.module.hnet,
                target_network=method.module.target_network,
                evaluation_dataset="validation",
                epsilon=method.current_epsilon,
                number_of_task=task_id,
                device=device
            )

            no_incorrect = calculate_no_incorrectly_classified_hypercubes(worst_case_prediction, gt_output)
            log_metrics(iteration, task_id, loss, accuracy, int(no_incorrect), method.current_epsilon)

            if should_update_best(accuracy, best_val_accuracy, iteration, total_no_iterations):
                log.info("New best val acc")
                best_val_accuracy = accuracy
                best_hnet = deepcopy(method.module.hnet)
                best_target_network = deepcopy(method.module.target_network)

            maybe_step_scheduler(method, iteration, total_no_iterations, accuracy, no_epochs)

    return best_hnet, best_target_network

def calculate_no_incorrectly_classified_hypercubes(worst_case_prediction: torch.Tensor, gt_output: torch.Tensor) -> float:
    """
    Calculates the number of incorrectly classified hypercubes.
    """
    return (worst_case_prediction.argmax(dim=1) != gt_output).float().sum().item()

def calculate_number_of_iterations(number_of_samples: int, batch_size: int, number_of_epochs: int) -> Tuple[int, int]:
    """
    Calculates the number of iterations per epoch and the total number of iterations for training.
    """
    no_of_iterations_per_epoch: int = int(np.ceil(number_of_samples / batch_size))
    total_no_of_iterations: int = no_of_iterations_per_epoch * number_of_epochs
    return no_of_iterations_per_epoch, total_no_of_iterations

def calculate_accuracy(data, hnet: torch.nn.Module, target_network: torch.nn.Module, evaluation_dataset: str,
                        epsilon: float, number_of_task: int, device: torch.device) -> float:
    """
    Calculates the classification accuracy of a neural network model on a specified evaluation dataset.
    """
    hnet.eval()
    target_network.eval()

    if evaluation_dataset == "validation":
        input_data = data.get_val_inputs()
        output_data = data.get_val_outputs()
    elif evaluation_dataset == "test":
        input_data = data.get_test_inputs()
        output_data = data.get_test_outputs()

    test_input = data.input_to_torch_tensor(input_data, device, mode="inference")
    test_output = data.output_to_torch_tensor(output_data, device, mode="inference")
    gt_classes = test_output.max(dim=1)[1]

    target_weights = hnet.forward(cond_id=number_of_task)
    logits, _ =  target_network.forward(
            test_input,
            epsilon=epsilon, 
            weights=target_weights
        )
    predictions = logits.max(dim=1)[1]
    accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0

    return accuracy.item()

def evaluate_previous_tasks(hnet: torch.nn.Module, target_network: torch.nn.Module, dataframe: pd.DataFrame, task_id: int, 
                            task_datasets: Iterable, epsilon: float, device: torch.device) -> pd.DataFrame:
    """
    Evaluates the performance of a continual learning method on all previously learned tasks.
    """
    for current_task_id in range(task_id + 1):
        currently_tested_task = task_datasets[current_task_id]

        accuracy = calculate_accuracy(
            data=currently_tested_task,
            hnet=hnet,
            target_network=target_network,
            evaluation_dataset="test",
            epsilon=epsilon,
            number_of_task=current_task_id,
            device=device
        )

        result = {
            "after_learning_of_task": task_id,
            "tested_task": current_task_id,
            "accuracy": accuracy,
        }
        log.info(f"Accuracy for task {current_task_id}: {accuracy:.5f}%")
        dataframe = pd.concat([dataframe, pd.DataFrame([result])], ignore_index=True)

    return dataframe

def calculate_backward_transfer(dataframe: pd.DataFrame) -> float:
    """
    Calculates the backward transfer metric (BWT).
    """
    task_ids = sorted(dataframe["after_learning_of_task"].unique())
    bwt = 0.0
    for task_id in task_ids[:-1]:
        last_acc = dataframe[(dataframe["after_learning_of_task"] == max(task_ids)) & (dataframe["tested_task"] == task_id)]["accuracy"].values[0]
        initial_acc = dataframe[(dataframe["after_learning_of_task"] == task_id) & (dataframe["tested_task"] == task_id)]["accuracy"].values[0]
        bwt += (last_acc - initial_acc)
    return bwt / (len(task_ids) - 1)

def plot_accuracy_progression(all_accuracies: List[List[float]], save_path: str) -> None:
    """
    Plots the test accuracy after each task for all tasks learned so far.
    Includes:
      - Accuracy just after learning each task
      - Accuracy of each task after all tasks are learned
    """

    # Convert list of task accuracies to uniform array with NaNs
    all_accuracies = np.array([
        np.pad(task_acc, (0, len(all_accuracies) - len(task_acc)), constant_values=np.nan)
        for task_acc in all_accuracies
    ])  # shape: (num_tasks_learned, num_tasks_total)

    num_tasks = all_accuracies.shape[1]
    tasks = np.arange(1, num_tasks + 1)

    # Extract just-after-training and after-all-training accuracy
    just_after_training = np.diag(all_accuracies)  # accuracy[i][i]
    after_all_training = all_accuracies[-1]        # last row

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, just_after_training, marker='o', label="Just After Training")
    plt.plot(tasks, after_all_training, marker='s', label="After All Tasks Trained")

    plt.xlabel("Number of Task", fontsize=14)
    plt.ylabel("Accuracy [%]", fontsize=14)
    plt.title("Test Accuracy Progression", fontsize=16)
    plt.xticks(ticks=tasks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    if wandb.run:
        wandb.log({"accuracy_progression": wandb.Image(save_path)})
