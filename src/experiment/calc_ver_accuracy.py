import os
import torch
import logging
import matplotlib.pyplot as plt
import wandb
from typing import List, Optional
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.fabric import setup_fabric
from model.model_abc import CLModuleABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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


def compute_verified_accuracy_per_task(
    model: CLModuleABC,
    datasets: List,
    fabric,
    config: DictConfig,
) -> List[float]:
    """
    Computes verified accuracy for each task in the dataset.

    Verified accuracy means the fraction of inputs where the predicted class
    remains the same under IBP bounds (lower == upper == true label).

    Args:
        model (CLModuleABC): The model to evaluate.
        datasets (List): List of task datasets.
        fabric: Fabric device and setup handler.
        config (DictConfig): Configuration object with experiment params.

    Returns:
        List[float]: Verified accuracy for each task.
    """
    accuracies = []
    for task_id, dataset in enumerate(datasets):
        inputs, targets = dataset.get_test_inputs(), dataset.get_test_outputs()
        test_input = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
        test_target = dataset.output_to_torch_tensor(targets, fabric.device)

        if test_target.ndim > 1 and test_target.size(1) > 1:
            test_target = test_target.argmax(dim=1)

        with torch.no_grad():
            logits, eps = model(x=test_input, epsilon=config.exp.epsilon, task_id=task_id)
            lower = logits - eps
            upper = logits + eps

            lower_pred = lower.argmax(dim=-1)
            upper_pred = upper.argmax(dim=-1)

            verified = (lower_pred == upper_pred) & (lower_pred == test_target)
            verified_acc = verified.float().mean().item()
            accuracies.append(verified_acc)

    return accuracies


def plot_per_task_verified_accuracy(
    acc_nomixup: List[float],
    acc_mixup: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a grouped bar chart of verified accuracy per task for No Mixup and Mixup models.

    Args:
        acc_nomixup (List[float]): Verified accuracy list for No Mixup model.
        acc_mixup (List[float]): Verified accuracy list for Mixup model.
        save_path (Optional[str]): Path to save the figure. If None, shows the plot.
    """
    tasks = list(range(len(acc_nomixup)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(tasks, acc_nomixup, width=width, label='No Mixup', color='skyblue')
    plt.bar([t + width for t in tasks], acc_mixup, width=width, label='Mixup', color='salmon')
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Verified Accuracy', fontsize=12)
    plt.title('Per-Task Verified Accuracy: Mixup vs No Mixup', fontsize=14)
    plt.xticks([t + width/2 for t in tasks], tasks)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def experiment(config: DictConfig) -> None:
    """
    Main experiment function: loads dataset and models, computes per-task verified accuracy
    for both No Mixup and Mixup models, then plots and logs the results.

    Args:
        config (DictConfig): Configuration object with all experiment parameters.
    """
    number_of_tasks = config.dataset.number_of_tasks

    log.info("Preparing datasets")
    cl_dataset = instantiate(config.dataset)
    task_datasets = cl_dataset.prepare_tasks(os.getenv("DATA_DIR"))

    log.info("Setting up Fabric")
    fabric = setup_fabric(config)

    log.info("Initializing models")
    model = fabric.setup(instantiate(config.model, number_of_tasks=number_of_tasks))
    mixup_model = fabric.setup(instantiate(config.model, number_of_tasks=number_of_tasks))

    log.info("Loading weights")
    hnet_weights = torch.load(f"{config.exp.path_to_weights}/hnet.pt", map_location=fabric.device)
    hnet_mixup_weights = torch.load(f"{config.exp.path_to_mixup_weights}/hnet.pt", map_location=fabric.device)

    model.hnet.load_state_dict(prepare_weights(hnet_weights, model), strict=False)
    mixup_model.hnet.load_state_dict(prepare_weights(hnet_mixup_weights, mixup_model), strict=False)
    model.hnet.eval()
    mixup_model.hnet.eval()

    log.info("🔍 Computing per-task verified accuracy")
    acc_nomixup = compute_verified_accuracy_per_task(model, task_datasets, fabric, config)
    acc_mixup = compute_verified_accuracy_per_task(mixup_model, task_datasets, fabric, config)

    log.info(f"Per-task verified accuracy No Mixup: {acc_nomixup}")
    log.info(f"Per-task verified accuracy Mixup:    {acc_mixup}")

    save_path = os.path.join(config.exp.log_dir, "per_task_verified_accuracy.png")
    plot_per_task_verified_accuracy(acc_nomixup, acc_mixup, save_path)

    if wandb.run:
        wandb.log({"per_task_verified_accuracy": wandb.Image(save_path)})

    log.info(f"Saved per-task verified accuracy plot at: {save_path}")
