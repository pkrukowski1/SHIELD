import os
import torch
import logging
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt

from utils.fabric import setup_fabric
from utils.handy_functions import prepare_weights

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def experiment(config: DictConfig) -> None:
    """
    Runs FGSM attacks with increasing perturbation sizes and evaluates average accuracy.

    Args:
        config (DictConfig): Hydra config with experiment setup.
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
    hnet_weights = torch.load(f"{config.exp.path_to_hnet_weights}/hnet.pt", map_location=fabric.device)
    hnet_mixup_weights = torch.load(f"{config.exp.path_to_mixup_weights}/hnet.pt", map_location=fabric.device)

    model.hnet.load_state_dict(prepare_weights(hnet_weights, model), strict=False)
    mixup_model.hnet.load_state_dict(prepare_weights(hnet_mixup_weights, mixup_model), strict=False)
    model.hnet.eval()
    mixup_model.hnet.eval()

    # Define range of epsilon values
    epsilons = np.linspace(0.0, config.exp.max_epsilon, num=config.exp.no_splits)
    all_results = []

    for epsilon in epsilons:
        log.info(f"Running FGSM attack with epsilon = {epsilon:.3f}")
        for model_type, current_model in [("standard", model), ("mixup", mixup_model)]:
            results = []

            for task_id, dataset in enumerate(task_datasets):
                # Prepare inputs and targets
                inputs = dataset.get_test_inputs()
                targets = dataset.get_test_outputs()
                test_input = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
                test_target = dataset.output_to_torch_tensor(targets, fabric.device, mode="inference")
                test_target = test_target.max(dim=1)[1]

                # Modify attack config dynamically
                attack_cfg = OmegaConf.to_container(config.exp.attack.fgsm, resolve=True)
                attack_cfg["eps"] = float(epsilon)
                attack_cfg = OmegaConf.create(attack_cfg)

                # Instantiate the attack
                attack = instantiate(attack_cfg, model=current_model, task_id=task_id, device=fabric.device)

                # Generate adversarial examples
                adv_input = attack.forward(test_input, test_target)

                # Compute accuracy on adversarial inputs
                with torch.no_grad():
                    logits, _ = current_model(x=adv_input, task_id=task_id, epsilon=epsilon)
                    preds = logits.max(dim=1)[1]
                    acc = 100.0 * (preds == test_target).float().mean().item()

                log.info(f"[{model_type.upper()} | epsilon={epsilon:.3f}] Task {task_id} Accuracy: {acc:.4f}%")
                results.append(acc)

            avg_acc = sum(results) / len(results)
            all_results.append({"epsilon": epsilon, "model": model_type, "avg_accuracy": avg_acc})
            log.info(f"[{model_type.upper()} | epsilon={epsilon:.3f}] Average Accuracy: {avg_acc:.4f}%")

    # Create DataFrame with all results
    df_all = pd.DataFrame(all_results)
    csv_path = os.path.join(config.exp.log_dir, "fgsm_accuracy_comparison.csv")
    df_all.to_csv(csv_path, sep=";", index=False)
    log.info(f"Saved combined CSV to {csv_path}")

    # Plot results
    plt.figure(figsize=(8, 6))
    for model_type in df_all["model"].unique():
        df_model = df_all[df_all["model"] == model_type]
        plt.plot(df_model["epsilon"], df_model["avg_accuracy"], label=model_type.capitalize(), marker='o')

    plt.xlabel("FGSM Perturbation Size", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.title("Adversarial Accuracy vs. FGSM Epsilon", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(config.exp.log_dir, "fgsm_accuracy_comparison.png")
    plt.savefig(plot_path)
    log.info(f"Saved accuracy plot to {plot_path}")