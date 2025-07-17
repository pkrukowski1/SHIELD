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

            # Instantiate the attack
            attack = instantiate(attack_cfg, model=model, device=fabric.device)

            # Generate adversarial examples
            adv_input = attack.forward(test_input, test_target, task_id=task_id)

            # Compute classical accuracy on adversarial inputs
            with torch.no_grad():
                logits, _ = model(x=adv_input, task_id=task_id, epsilon=config.method.epsilon)
                preds = logits.max(dim=1)[1]
                acc = 100.0 * (preds == test_target).float().mean().item()

            log.info(f"[{attack_name}] Task {task_id} | Adversarial Accuracy: {acc:.4f}%")
            results.append({"task_id": task_id, "adversarial_accuracy": acc})

        # Save results to CSV
        results_df = pd.DataFrame(results)
        out_path = os.path.join(config.exp.log_dir, f"adversarial_accuracy_{attack_name}.csv")
        results_df.to_csv(out_path, sep=";", index=False)
        log.info(f"Saved results for {attack_name} to: {out_path}")
