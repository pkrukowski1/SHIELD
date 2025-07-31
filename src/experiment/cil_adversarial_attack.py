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
                           task_tensor: torch.Tensor, num_tasks: int, device: torch.device, epsilon: float
                           ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Performs task inference using minimum entropy and computes accuracy
    only when the task is correctly inferred. Returns predictions, accuracy,
    inferred task IDs, and a mask indicating which samples had correct task inference.
    """
    with torch.no_grad():
        logits_all = []
        entropy_all = []

        for tid in range(num_tasks):
            logits, _ = model(x=input_tensor, task_id=tid, epsilon=epsilon)
            prob = F.softmax(logits, dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
            logits_all.append(logits)
            entropy_all.append(entropy)

        logits_all = torch.stack(logits_all)       # [num_tasks, batch, num_classes]
        entropy_all = torch.stack(entropy_all)     # [num_tasks, batch]

        inferred_task_ids = entropy_all.argmin(dim=0)        # [batch]
        correct_task_mask = (inferred_task_ids == task_tensor)

        preds = torch.full_like(target_tensor, fill_value=-1, device=device)
        correct = 0
        total = input_tensor.size(0)

        for i in range(total):
            if correct_task_mask[i]:
                pred_logits = logits_all[inferred_task_ids[i], i]
                preds[i] = pred_logits.argmax()
                if preds[i] == target_tensor[i]:
                    correct += 1
            # else: zero accuracy by default

        accuracy = 100.0 * correct / total
        return preds, accuracy, inferred_task_ids, correct_task_mask


def experiment(config: DictConfig) -> None:
    """
    Runs all defined adversarial attacks and computes clean and adversarial accuracy
    under the Class-Incremental Learning (CIL) setting, where task identity is unknown.
    Accuracy is only counted when task inference is correct.
    Adversarial accuracy is computed only for clean-correct predictions with correct task.
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
            task_tensor = torch.full_like(test_target, fill_value=task_id)

            # ----- CLEAN INFERENCE (CIL) -----
            clean_preds, clean_acc, _, correct_task_mask = infer_predictions_cil(
                model=model,
                input_tensor=test_input,
                target_tensor=test_target,
                task_tensor=task_tensor,
                num_tasks=number_of_tasks,
                device=fabric.device,
                epsilon=config.exp.epsilon
            )

            clean_correct_mask = (clean_preds == test_target) & correct_task_mask

            # ----- ADVERSARIAL INFERENCE (CIL) -----
            if clean_correct_mask.sum().item() == 0:
                adv_acc = 0.0
            else:
                filtered_input = test_input[clean_correct_mask]
                filtered_target = test_target[clean_correct_mask]
                filtered_task = task_tensor[clean_correct_mask]

                attack = instantiate(attack_cfg, model=model, task_id=task_id, device=fabric.device)
                adv_input = attack.forward(filtered_input, filtered_target)

                _, adv_acc_filtered, _, _ = infer_predictions_cil(
                    model=model,
                    input_tensor=adv_input,
                    target_tensor=filtered_target,
                    task_tensor=filtered_task,
                    num_tasks=number_of_tasks,
                    device=fabric.device,
                    epsilon=config.exp.epsilon
                )

                adv_acc = adv_acc_filtered * (filtered_input.size(0) / test_input.size(0))

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

        # Log final results to console
        log.info(f"[{attack_name}] FINAL AVERAGE | Clean Acc: {avg_clean:.2f}%, Adversarial Acc: {avg_adv:.2f}%")

        # Save CSV
        results_df = pd.DataFrame(results)
        out_path = os.path.join(config.exp.log_dir, f"cil_accuracy_{attack_name}.csv")
        results_df.to_csv(out_path, sep=";", index=False)
        log.info(f"Saved clean and adversarial CIL results for {attack_name} to: {out_path}")
