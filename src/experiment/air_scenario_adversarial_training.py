import pandas as pd
import os
from copy import deepcopy

from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
import logging

from utils.fabric import setup_fabric
from utils.handy_functions import write_pickle_file, plot_heatmap
from .interval_training import evaluate_previous_tasks, calculate_backward_transfer, plot_accuracy_progression, train_single_task

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def experiment(config: DictConfig) -> None:
    """
    Executes the Continual Learning experiment with Adversarial Curriculum support.
    
    Loads configuration from YAML via Hydra, initializes the 'SHIELDWithAdvAttacks'
    method, and iterates through the task sequence.
    """

    number_of_tasks = config.dataset.number_of_tasks
    log.info(f"Preparing the whole dataset")
    
    cl_dataset = instantiate(config.dataset, number_of_tasks=1)
    task_datasets = cl_dataset.prepare_tasks(os.getenv("DATA_DIR"))

    log.info('Launching Fabric')
    fabric = setup_fabric(config)

    log.info('Building model')
    model = fabric.setup(instantiate(config.model, number_of_tasks=number_of_tasks))

    log.info('Setting up method')
    method = instantiate(
        config.method, 
        module=model
    )

    # Logging setup
    if hasattr(method, 'scenario'):
        log.info(f"Adversarial Scenario Active: {method.scenario.value}")
    
    dataframe = pd.DataFrame(columns=["after_learning_of_task", "tested_task", "accuracy"])
    all_accuracies = []
    
    try:
        for task_id in range(number_of_tasks):

            log.info(f"--- Starting Task {task_id + 1}/{number_of_tasks} ---")

            if hasattr(method, '_determine_attack_type'):
                attack_type = method._determine_attack_type(task_id)
                log.info(f"Attack Strategy for Task {task_id}: {attack_type.name}")

            method.setup_task(task_id)

            best_module = train_single_task(
                method=method,
                task_id=task_id,
                task_datasets=task_datasets,
                config=config,
                device=fabric.device
            )
           
            method.module.load_state_dict(deepcopy(best_module.state_dict()))
            
            method.detach_embedding(task_id)

            dataframe = evaluate_previous_tasks(
                module=best_module,
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

            accs_up_to_now = dataframe[dataframe["after_learning_of_task"] == task_id]["accuracy"]
            avg_acc = accs_up_to_now.mean()
            log.info(f"Average accuracy after task {task_id}: {avg_acc:.2f}%")
            
            if wandb.run:
                wandb.log({f"avg_accuracy_up_to_task_{task_id}": avg_acc})

            if config.exp.save_model_per_task:
                write_pickle_file(f'{config.exp.log_dir}/hnet_after_{task_id+1}_task', best_module.hnet.weights)

    except IndexError:
        log.error("IndexError during training loop. Check dataset splitting.")
        pass

    if not config.exp.save_model_per_task:
        write_pickle_file(f'{config.exp.log_dir}/hnet', best_module.hnet.weights)

    plot_heatmap(f'{config.exp.log_dir}/results.csv')

    if task_id > 0:
        bwt = calculate_backward_transfer(dataframe)
        log.info(f"Backward transfer: {bwt:.5f}")
        if wandb.run:
            wandb.log({"backward_transfer": bwt})

    plot_accuracy_progression(all_accuracies, f"{config.exp.log_dir}/accuracy_progression.png")
