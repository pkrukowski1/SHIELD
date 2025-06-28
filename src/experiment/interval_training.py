import logging

from copy import deepcopy
import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch

import wandb

from typing import Tuple, Iterable, Optional

from utils.fabric import setup_fabric
from utils.handy_functions import calculate_accuracy
from method.method_abc import MethodABC
 

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
    

def experiment(config: DictConfig):
    """
    Full training and testing on given scenario.
    """

    if config.exp.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    calc_bwt = getattr(config.exp, 'calc_bwt', False)
    calc_fwt = getattr(config.exp, 'calc_fwt', False)
    acc_table = getattr(config.exp, 'acc_table', False)
    stop_task = getattr(config.exp, 'stop_after_task', None)
    save_model = hasattr(config.exp, 'model_path')

    no_tasks = config.exp.no_tasks

    log.info(f'Initializing scenarios')
    cl_dataset = instantiate(config.dataset)

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.model))

    log.info(f'Setting up method')
    method = instantiate(config.method)(model)

    for task_id in no_tasks:
        train_single_task(
            method=method,
            task_id=task_id,
            cl_dataset=cl_dataset,
            config=config
        )

def should_log(
    iteration: int,
    total_no_iterations: int,
    no_epochs: Optional[int],
    no_iterations_per_epoch: Optional[int]
) -> bool:
    """
    Determine if logging or validation should be performed at this iteration.
    """
    return (
        (iteration % 100 == 0)
        or (iteration == total_no_iterations - 1)
        or (no_epochs is not None and ((iteration + 1) % no_iterations_per_epoch == 0))
    )

def maybe_log_epoch(
    iteration: int,
    no_epochs: Optional[int],
    no_iterations_per_epoch: Optional[int]
) -> None:
    """
    Optionally log the current epoch number.
    """
    if no_epochs is not None and no_iterations_per_epoch:
        current_epoch = (iteration + 1) // no_iterations_per_epoch
        log.info(f"Current epoch: {current_epoch}")

def log_metrics(
    iteration: int,
    task_id: int,
    loss: torch.Tensor,
    accuracy: float,
    no_incorrect_hypercubes: int
) -> None:
    """
    Log current training loss and validation metrics.
    """
    log.info(
        f"Task {task_id}, iteration: {iteration + 1},"
        f" train loss: {loss.item()}, validation accuracy: {accuracy},"
        f" no incorrectly classified hypercubes: {no_incorrect_hypercubes}"
    )

def should_update_best(
    accuracy: float,
    best_val_accuracy: float,
    iteration: int,
    total_no_iterations: int
) -> bool:
    """
    Check if the current model is the best so far.
    """
    return accuracy > best_val_accuracy and iteration >= total_no_iterations // 2

def maybe_step_scheduler(
    method: "MethodABC",
    iteration: int,
    total_no_iterations: int,
    accuracy: float,
    no_epochs: Optional[int]
) -> None:
    """
    Step the learning rate scheduler if applicable.
    """
    if (
        no_epochs is not None
        and method.scheduler is not None
        and ((iteration + 1) % total_no_iterations == 0)
    ):
        print("Finishing the current epoch")
        method.make_scheduler_step(accuracy)

def train_single_task(
    method: "MethodABC", 
    task_id: int, 
    cl_dataset: Iterable,
    config: DictConfig,
    device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Train the model for a single task.

    Args:
        method (MethodABC): The continual learning method instance with model and optimizer.
        task_id (int): Index of the task to train.
        cl_dataset (Iterable): Dataset containing task-specific data objects.
        config (DictConfig): Training configuration with fields like `no_epochs`, `no_iterations`, `batch_size`.
        device (torch.device): Target device for training (CPU or CUDA).

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: Best performing hypernetwork and target network models.
    """
    no_iterations: Optional[int] = getattr(config, "no_iterations", None)
    no_epochs: Optional[int] = getattr(config, "no_epochs", None)
    batch_size: int = config.batch_size

    assert no_iterations is not None or no_epochs is not None, \
        "Arguments `no_iterations` and `no_epochs` cannot both be `None`"

    method.setup_optim()
    current_dataset_instance = cl_dataset[task_id]

    if no_epochs is not None:
        no_iterations_per_epoch, total_no_iterations = calculate_number_of_iterations(
            number_of_samples=current_dataset_instance.num_train_samples,
            batch_size=batch_size,
            number_of_epochs=no_epochs
        )
    else:
        no_iterations_per_epoch = None
        total_no_iterations = no_iterations

    best_hypernetwork = deepcopy(method.module.hnet)
    best_target_network = deepcopy(method.module.target_network)
    best_val_accuracy = 0.0

    method.module.hnet.train()
    log.info(f"Train the {task_id}-th task")

    for iteration in range(total_no_iterations):
        current_batch = current_dataset_instance.next_train_batch(batch_size)

        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], device, mode="train"
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], device, mode="train"
        )

        loss, worst_case_prediction = method.forward(tensor_input, tensor_output, task_id)
        loss = loss.mean()
        method.backward(loss)

        if should_log(iteration, total_no_iterations, no_epochs, no_iterations_per_epoch):
            maybe_log_epoch(iteration, no_epochs, no_iterations_per_epoch)

            accuracy = calculate_accuracy(
                current_dataset_instance,
                model=method.module,
                evaluation_dataset="validation",
                epsilon=method.current_epsilon,
                number_of_task=task_id,
                device=device
            )

            no_incorrect = calculate_no_incorrectly_classified_hypercubes(worst_case_prediction)
            log_metrics(iteration, task_id, loss, accuracy, no_incorrect)

            if should_update_best(accuracy, best_val_accuracy, iteration, total_no_iterations):
                log.info("New best val acc")
                best_val_accuracy = accuracy
                best_hypernetwork = deepcopy(method.module.hnet)
                best_target_network = deepcopy(method.module.target_network)

            maybe_step_scheduler(method, iteration, total_no_iterations, accuracy, no_epochs)

    return best_hypernetwork, best_target_network




def calculate_no_incorrectly_classified_hypercubes(worst_case_prediction: torch.Tensor,
                                                    gt_output: torch.Tensor) -> float:
    """
    Calculates the number of incorrectly classified hypercubes.
    Given the worst-case predictions and ground truth outputs, this function computes
    how many predictions do not match the ground truth labels.
    Args:
        worst_case_prediction (torch.Tensor): The predicted outputs for each hypercube,
            typically of shape (N, C), where N is the number of hypercubes and C is the number of classes.
        gt_output (torch.Tensor): The ground truth class labels for each hypercube, of shape (N,).
    Returns:
        float: The number of hypercubes that are incorrectly classified.
    """
    
    return (worst_case_prediction.argmax(dim=1) != gt_output).float().sum().item()
        
def calculate_number_of_iterations(number_of_samples: int, batch_size: int, number_of_epochs: int) -> Tuple[int,int]:
    """
    Calculates the number of iterations per epoch and the total number of iterations for training.

    Args:
        number_of_samples (int): The total number of samples in the dataset.
        batch_size (int): The number of samples per batch.
        number_of_epochs (int): The total number of epochs for training.

    Returns:
        Tuple[int, int]:
            - The number of iterations per epoch.
            - The total number of iterations across all epochs.
    """
    
    no_of_iterations_per_epoch = int(np.ceil(number_of_samples / batch_size))
    total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
    return no_of_iterations_per_epoch, total_no_of_iterations


def build_multiple_task_experiment(dataset_list_of_tasks, parameters):

    dataframe = pd.DataFrame(columns=["after_learning_of_task", "tested_task", "accuracy"])

    use_batch_norm_memory = parameters["use_batch_norm"]
    hypernetwork.train()
    for no_of_task in range(parameters["number_of_tasks"]):
        hypernetwork, target_network = train_single_task(
            hypernetwork,
            target_network,
            criterion,
            parameters,
            dataset_list_of_tasks,
            no_of_task,
        )
        hypernetwork.conditional_params[no_of_task].requires_grad_(False)
        if no_of_task == (parameters["number_of_tasks"] - 1):
            write_pickle_file(
                f'{parameters["saving_folder"]}/hnet_{parameters["perturbation_epsilon"]}', hypernetwork.weights
            )
        dataframe = evaluate_previous_tasks(
            hypernetwork,
            target_network,
            dataframe,
            dataset_list_of_tasks,
            parameters={
                "device": parameters["device"],
                "use_batch_norm_memory": use_batch_norm_memory,
                "number_of_task": no_of_task,
                "perturbation_epsilon": parameters["perturbation_epsilon"]
            },
        )
        dataframe = dataframe.astype({"after_learning_of_task": "int", "tested_task": "int"})
        dataframe.to_csv(f'{parameters["saving_folder"]}/results_{parameters["dataset"]}.csv', sep=";")
    return hypernetwork, target_network, dataframe