import os
import torch
import logging
import pandas as pd
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.fabric import setup_fabric
from utils.handy_functions import prepare_weights

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def experiment(config: DictConfig) -> None:
    """
    Runs all defined adversarial attacks and computes classical accuracy
    on perturbed test inputs, saving the results per attack.

    Args:
        config (DictConfig): Hydra config with experiment setup.
    """
    number_of_tasks = config.dataset.number_of_tasks

    log.info("Preparing datasets")
    cl_dataset = instantiate(config.dataset)
    task_datasets = cl_dataset.prepare_tasks(os.getenv("DATA_DIR"))

    log.info("Setting up Fabric")
    fabric = setup_fabric(config)

    log.info("Initializing model")
    model = fabric.setup(instantiate(config.model, number_of_tasks=number_of_tasks))

    log.info("Loading model weights")
    hnet_weights = torch.load(f"{config.exp.path_to_weights}/hnet.pt", map_location=fabric.device)
    model.hnet.load_state_dict(prepare_weights(hnet_weights, model), strict=False)
    model.hnet.eval()

    # Go through each attack defined in config
    for attack_name, attack_cfg in config.exp.attack.items():
        log.info(f"Running attack: {attack_name}")
        results = []

        for task_id, dataset in enumerate(task_datasets):

            # Prepare inputs and targets
            inputs = dataset.get_test_inputs()
            targets = dataset.get_test_outputs()
            test_input = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
            test_target = dataset.output_to_torch_tensor(targets, fabric.device, mode="inference")
            test_target = test_target.max(dim=1)[1]

            assert test_input.ndim == 2, f"Expected flattened input [N, F], got shape {test_input.shape}"
            N, F = test_input.shape

            num_channels = len(config.exp.standarization.mean)
            assert F % num_channels == 0, (
                f"Input feature size {F} not divisible by channel count {num_channels}"
            )

            pixels_per_channel = F // num_channels

            mean_vals = torch.tensor(config.exp.standarization.mean, device=fabric.device, dtype=test_input.dtype)
            std_vals = torch.tensor(config.exp.standarization.std, device=fabric.device, dtype=test_input.dtype)

            mean_flat = mean_vals.repeat_interleave(pixels_per_channel)  # shape [F]
            std_flat = std_vals.repeat_interleave(pixels_per_channel)    # shape [F]

            mean = mean_flat.unsqueeze(0).expand_as(test_input)  # [N, F]
            std = std_flat.unsqueeze(0).expand_as(test_input)    # [N, F]

            test_input = (test_input - mean) / std
            min_val, max_val = test_input.min().float(), test_input.max().float()

            # Instantiate the attack
            attack = instantiate(attack_cfg, model=model, task_id=task_id, device=fabric.device)

            correct = 0
            total = 0
            batch_size = 1000

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_input = test_input[start:end]
                batch_target = test_target[start:end]

                adv_input = attack.forward(batch_input, batch_target, min_val=min_val, max_val=max_val)

                # Compute classical accuracy on adversarial inputs
                with torch.no_grad():
                    logits, _ = model(x=adv_input, task_id=task_id, epsilon=config.exp.epsilon)
                    preds = logits.max(dim=1)[1]
                    correct += (preds == batch_target).sum().item()
                    total += batch_target.size(0)

            acc = 100.0 * (correct / total) if total > 0 else 0.0

            log.info(f"[{attack_name}] Task {task_id} | Adversarial Accuracy: {acc:.4f}%")
            results.append({"task_id": task_id, "adversarial_accuracy": acc})

        # Compute average accuracy across all tasks
        avg_acc = sum(r["adversarial_accuracy"] for r in results) / len(results)
        log.info(f"[{attack_name}] Average Adversarial Accuracy over all tasks: {avg_acc:.4f}%")

        # Append average to results
        results.append({"task_id": "average", "adversarial_accuracy": avg_acc})

        # Save results to CSV
        results_df = pd.DataFrame(results)
        out_path = os.path.join(config.exp.log_dir, f"adversarial_accuracy_{attack_name}.csv")
        results_df.to_csv(out_path, sep=";", index=False)
        log.info(f"Saved results for {attack_name} to: {out_path}")
