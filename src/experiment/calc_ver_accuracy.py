import os
import torch
import logging
import matplotlib.pyplot as plt
import wandb
from typing import Dict
from collections import defaultdict
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.fabric import setup_fabric
from model.model_abc import CLModuleABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def prepare_weights(hnet_weights: dict, model: CLModuleABC) -> dict:
    return {
        hypernet_key: hnet_weights[loaded_key]
        for (hypernet_key, _), loaded_key in zip(model.hnet.named_parameters(), hnet_weights.keys())
    }


def compute_per_class_verified_accuracy(
    model: CLModuleABC,
    datasets,
    fabric,
    config,
    num_classes: int
) -> Dict[int, float]:
    verified_correct = defaultdict(int)
    total_per_class = defaultdict(int)

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

            for cls in range(num_classes):
                class_mask = (test_target == cls)
                verified_correct[cls] += (verified & class_mask).sum().item()
                total_per_class[cls] += class_mask.sum().item()

    return {
        cls: (verified_correct[cls] / total_per_class[cls]) if total_per_class[cls] > 0 else 0.0
        for cls in range(num_classes)
    }


def plot_verified_accuracy_bar(per_class1, per_class2, label1, label2, save_path=None):
    classes = sorted(per_class1.keys())
    acc1 = [per_class1[c] for c in classes]
    acc2 = [per_class2[c] for c in classes]

    x = range(len(classes))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x, acc1, width=width, label=label1, color='skyblue')
    plt.bar([i + width for i in x], acc2, width=width, label=label2, color='salmon')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Verified Accuracy", fontsize=12)
    plt.title("Per-Class Verified Accuracy: Mixup vs No Mixup", fontsize=14)
    plt.xticks([i + width / 2 for i in x], classes)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def experiment(config: DictConfig) -> None:
    number_of_tasks = config.dataset.number_of_tasks
    no_classes_per_task = config.model.no_classes_per_task

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

    log.info("🔍 Computing per-class verified accuracy")
    verified_acc_no_mixup = compute_per_class_verified_accuracy(model, task_datasets, fabric, config, no_classes_per_task)
    verified_acc_mixup = compute_per_class_verified_accuracy(mixup_model, task_datasets, fabric, config, no_classes_per_task)

    avg_no_mixup = sum(verified_acc_no_mixup.values()) / len(verified_acc_no_mixup)
    avg_mixup = sum(verified_acc_mixup.values()) / len(verified_acc_mixup)

    log.info(f"Avg Verified Accuracy (No Mixup): {avg_no_mixup:.4f}")
    log.info(f"Avg Verified Accuracy (Mixup):    {avg_mixup:.4f}")
    log.info(f"Per-Class Verified Accuracy (No Mixup): {verified_acc_no_mixup}")
    log.info(f"Per-Class Verified Accuracy (Mixup):    {verified_acc_mixup}")

    save_path = os.path.join(config.exp.log_dir, "per_class_verified_acc.png")
    plot_verified_accuracy_bar(
        verified_acc_no_mixup,
        verified_acc_mixup,
        label1="No Mixup",
        label2="Mixup",
        save_path=save_path
    )

    if wandb.run:
        wandb.log({
            "verified_accuracy_per_class_plot": wandb.Image(save_path),
            "avg_verified_accuracy_no_mixup": avg_no_mixup,
            "avg_verified_accuracy_mixup": avg_mixup,
            "verified_accuracy_per_class_no_mixup": verified_acc_no_mixup,
            "verified_accuracy_per_class_mixup": verified_acc_mixup
        })

    log.info(f"Saved per-class verified accuracy plot at: {save_path}")
