import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import pandas as pd
import os
from typing import Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.fabric import setup_fabric
from utils.handy_functions import prepare_weights

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def infer_predictions_cil(model: nn.Module, input_tensor: torch.Tensor, target_tensor: torch.Tensor,
                           num_tasks: int, device: torch.device, epsilon: float) -> Tuple[torch.Tensor, float]:
    """
    Performs task-inference using minimum entropy and returns predicted labels and accuracy.
    """
    with torch.no_grad():
        logits_all = []
        entropy_all = []

        for tid in range(num_tasks):
            logits, _ = model(x=input_tensor, task_id=tid, epsilon=epsilon)
            prob = F.softmax(logits, dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # avoid log(0)
            logits_all.append(logits)
            entropy_all.append(entropy)

        logits_all = torch.stack(logits_all)
        entropy_all = torch.stack(entropy_all)

        inferred_task_ids = entropy_all.argmin(dim=0)

        preds = torch.empty_like(target_tensor, device=device)
        for i in range(input_tensor.size(0)):
            pred_logits = logits_all[inferred_task_ids[i], i]
            preds[i] = pred_logits.argmax()

        accuracy = 100.0 * (preds == target_tensor).float().mean().item()
    return preds, accuracy

def experiment(config: DictConfig) -> None:
    """
    Runs all defined adversarial attacks and computes clean and adversarial accuracy
    under the Class-Incremental Learning (CIL) setting, where task identity is unknown.
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

    for attack_name, attack_cfg in config.exp.attack.items():
        log.info(f"Running attack: {attack_name}")
        results = []

        for task_id, dataset in enumerate(task_datasets):
            # Prepare test inputs and labels
            inputs = dataset.get_test_inputs()
            targets = dataset.get_test_outputs()
            test_input = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
            test_target = dataset.output_to_torch_tensor(targets, fabric.device, mode="inference")
            test_target = test_target.max(dim=1)[1]

            # ----- CLEAN INFERENCE (CIL) -----
            _, clean_acc = infer_predictions_cil(
                model=model,
                input_tensor=test_input,
                target_tensor=test_target,
                num_tasks=number_of_tasks,
                device=fabric.device,
                epsilon=config.exp.epsilon
            )

            # ----- ADVERSARIAL INFERENCE (CIL) -----
            attack = instantiate(attack_cfg, model=model, task_id=task_id, device=fabric.device)
            adv_input = attack.forward(test_input, test_target)

            _, adv_acc = infer_predictions_cil(
                model=model,
                input_tensor=adv_input,
                target_tensor=test_target,
                num_tasks=number_of_tasks,
                device=fabric.device,
                epsilon=config.exp.epsilon
            )

            log.info(f"[{attack_name}] Task {task_id} | Clean Acc: {clean_acc:.2f}%, Adversarial Acc: {adv_acc:.2f}%")

            results.append({
                "task_id": task_id,
                "clean_accuracy": clean_acc,
                "adversarial_accuracy": adv_acc
            })

        # Compute and save averages
        avg_clean = sum(r["clean_accuracy"] for r in results) / len(results)
        avg_adv = sum(r["adversarial_accuracy"] for r in results) / len(results)
        results.append({
            "task_id": "average",
            "clean_accuracy": avg_clean,
            "adversarial_accuracy": avg_adv
        })

        # Save CSV
        results_df = pd.DataFrame(results)
        out_path = os.path.join(config.exp.log_dir, f"cil_accuracy_{attack_name}.csv")
        results_df.to_csv(out_path, sep=";", index=False)
        log.info(f"Saved clean and adversarial CIL results for {attack_name} to: {out_path}")
