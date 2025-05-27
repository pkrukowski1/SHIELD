import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
from copy import deepcopy
from retry import retry
import time

import torch
import torch.optim as optim
from torch import nn
from hypnettorch.hnets import HMLP
import hypnettorch.utils.hnet_regularizer as hreg


from IntervalNets.IntervalMLP import IntervalMLP
from IntervalNets.IntervalAlexNet import IntervalAlexNet
from IntervalNets.IntervalResNet18 import IntervalResNet18
from datasets import *


def get_shapes_of_network(model):
    shapes_of_model = []
    for layer in model.weights:
        shapes_of_model.append(list(layer.shape))
    return shapes_of_model


def calculate_number_of_iterations(number_of_samples, batch_size, number_of_epochs):
    no_of_iterations_per_epoch = int(np.ceil(number_of_samples / batch_size))
    total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
    return no_of_iterations_per_epoch, total_no_of_iterations


def calculate_accuracy(data, target_network, weights, parameters, evaluation_dataset):
    target_network = deepcopy(target_network)
    target_network.eval()
  
    if evaluation_dataset == "validation":
        input_data = data.get_val_inputs()
        output_data = data.get_val_outputs()
    elif evaluation_dataset == "test":
        input_data = data.get_test_inputs()
        output_data = data.get_test_outputs()

    test_input = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
    test_output = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")

    test_input.requires_grad = True
    gt_classes = test_output.max(dim=1)[1]

    if parameters["use_batch_norm_memory"]:
        logits = target_network.forward(
            test_input, 
            epsilon=parameters["perturbation_epsilon"],
            weights=weights, 
            condition=parameters["number_of_task"]
            )
    else:
        logits = target_network.forward(
            test_input, 
            epsilon=parameters["perturbation_epsilon"],
            weights=weights)

    # We take into the consideration only the midpoint of a hypercube 
    logits, _ = logits
    predictions = logits.max(dim=1)[1]

    accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0
    return accuracy


def evaluate_previous_tasks(hypernetwork, target_network, dataframe_results, list_of_permutations, parameters):
    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters["number_of_task"] + 1):
        currently_tested_task = list_of_permutations[task]
        hypernetwork_weights = hypernetwork.forward(cond_id=task)

        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork_weights,
            parameters=parameters,
            evaluation_dataset="test",
        )
        result = {
            "after_learning_of_task": parameters["number_of_task"],
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")
        dataframe_results = dataframe_results.append(result, ignore_index=True)
    return dataframe_results


def save_parameters(saving_folder, parameters, name=None):
    if name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"parameters_{current_time}.csv"
    with open(f"{saving_folder}/{name}", "w") as file:
        for key in parameters.keys():
            file.write(f"{key};{parameters[key]}\n")


def plot_heatmap(load_path):
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
    table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()


def train_single_task(hypernetwork, target_network, criterion, parameters, dataset_list_of_tasks, current_no_of_task):
    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam([*hypernetwork.parameters()],lr=parameters["learning_rate"])
    elif parameters["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop([*hypernetwork.parameters()], lr=parameters["learning_rate"])
    if parameters["best_model_selection_method"] == "val_loss":
        best_hypernetwork = deepcopy(hypernetwork)
        best_target_network = deepcopy(target_network)
        best_val_accuracy = 0.0
    hypernetwork.train()
    print(f"task: {current_no_of_task}")
    if current_no_of_task > 0:
        regularization_targets = hreg.get_current_targets(
            current_no_of_task, hypernetwork
        )
        previous_hnet_theta = None
        previous_hnet_embeddings = None
    use_batch_norm_memory = False
    current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
    if parameters["number_of_epochs"] is not None:
        (
            no_of_iterations_per_epoch,
            parameters["number_of_iterations"],
        ) = calculate_number_of_iterations(
            current_dataset_instance.num_train_samples,
            parameters["batch_size"],
            parameters["number_of_epochs"],
        )
        if parameters["lr_scheduler"]:
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "max",
                factor=np.sqrt(0.1),
                patience=5,
                min_lr=0.5e-6,
                cooldown=0,
                verbose=True,
            )
    # TODO save entry point
    for iteration in range(parameters["number_of_iterations"]):
        current_batch = current_dataset_instance.next_train_batch(
            parameters["batch_size"]
        )
        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], parameters["device"], mode="train"
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], parameters["device"], mode="train"
        )
        gt_output = tensor_output.max(dim=1)[1]

        optimizer.zero_grad()
        hnet_weights = hypernetwork.forward(cond_id=current_no_of_task)

        base_eps = parameters["perturbation_epsilon"]
        total_iterations = parameters["number_of_iterations"]
        inv_total_iterations = 1 / total_iterations
        if iteration <= total_iterations // 2:
            perturbation_epsilon = 2 * iteration * inv_total_iterations * base_eps
        else:
            perturbation_epsilon = base_eps

        prediction, eps_prediction = target_network.forward(
            tensor_input,
            epsilon=perturbation_epsilon, 
            weights=hnet_weights
        )

        z_lower = prediction - eps_prediction
        z_upper = prediction + eps_prediction
        z = torch.where((nn.functional.one_hot(gt_output, prediction.size(-1))).bool(), z_lower, z_upper)

        # To print only
        worst_case_error = (z.argmax(dim=1) != gt_output).float().sum().item()

        loss_spec = criterion(z, gt_output) # Worst-case loss
        loss_fit = criterion(prediction, gt_output) # Just cross-entropy loss
        if iteration <= total_iterations // 2:
            kappa = 1 - (iteration * inv_total_iterations)
        else:
            kappa = 0.5

        loss_current_task = kappa * loss_fit + (1 - kappa) * loss_spec
       
        loss_regularization = 0.0
        if current_no_of_task > 0:
            loss_regularization = hreg.calc_fix_target_reg(
                hypernetwork,
                current_no_of_task,
                targets=regularization_targets,
                mnet=target_network,
                prev_theta=previous_hnet_theta,
                prev_task_embs=previous_hnet_embeddings,
                inds_of_out_heads=None,
                batch_size=-1,
            )
        loss = (
            loss_current_task
            + parameters["beta"]
            * loss_regularization
            / max(1, current_no_of_task)
        )
        loss.backward()
        optimizer.step()

        if parameters["number_of_epochs"] is None:
            condition = (iteration % 100 == 0) or (
                iteration == (parameters["number_of_iterations"] - 1)
            )
        else:
            condition = (
                (iteration % 100 == 0)
                or (iteration == (parameters["number_of_iterations"] - 1))
                or (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            )

        if condition:
            if parameters["number_of_epochs"] is not None:
                current_epoch = (iteration + 1) // no_of_iterations_per_epoch
                print(f"Current epoch: {current_epoch}")
            accuracy = calculate_accuracy(
                current_dataset_instance,
                target_network,
                hnet_weights,
                parameters={
                    "device": parameters["device"],
                    "use_batch_norm_memory": use_batch_norm_memory,
                    "number_of_task": current_no_of_task,
                    "perturbation_epsilon": perturbation_epsilon
                },
                evaluation_dataset="validation",
            )
            print(
                f"Task {current_no_of_task}, iteration: {iteration + 1},"
                f" loss: {loss_current_task.item()}, validation accuracy: {accuracy},"
                f" No incorrectly classified hypercubes: {worst_case_error}"
            )
            if parameters["best_model_selection_method"] == "val_loss":

                # We need to ensure that the perturbation epsilon reached the maximum value
                if accuracy > best_val_accuracy and perturbation_epsilon == parameters["perturbation_epsilon"]:
                    print('new best val acc')
                    
                    best_val_accuracy = accuracy
                    best_hypernetwork = deepcopy(hypernetwork)
                    best_target_network = deepcopy(target_network)
            if (
                parameters["number_of_epochs"] is not None
                and parameters["lr_scheduler"]
                and (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            ):
                print("Finishing the current epoch")
                plateau_scheduler.step(accuracy)

    if parameters["best_model_selection_method"] == "val_loss":
        return best_hypernetwork, best_target_network
    else:
        return hypernetwork, target_network


def build_multiple_task_experiment(dataset_list_of_tasks, parameters):
    output_shape = list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]
    if parameters["target_network"] == "IntervalMLP":
        target_network = IntervalMLP(
            n_in=parameters["input_shape"],
            n_out=output_shape,
            hidden_layers=parameters["target_hidden_layers"],
            use_bias=parameters["use_bias"],
            no_weights=True,
        ).to(parameters["device"])
    elif parameters["target_network"] == "AlexNet":
         target_network = IntervalAlexNet(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            num_classes=output_shape,
            no_weights=True,
            use_batch_norm=parameters["use_batch_norm"],
            bn_track_stats=False,
            distill_bn_stats=False
        ).to(parameters["device"])
    elif parameters["target_network"] == "ResNet":
        target_network = IntervalResNet18(
                in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
                use_bias=True,
                use_fc_bias=parameters["use_bias"],
                bottleneck_blocks=False,
                num_classes=output_shape,
                num_feature_maps=[16, 32, 64, 128],
                blocks_per_group=[2, 2, 2, 2],
                no_weights=True,
                use_batch_norm=parameters["use_batch_norm"],
                projection_shortcut=True,
                bn_track_stats=False,
                cutout_mod=False,
                mode="default"
            ).to(parameters["device"])

    hypernetwork = HMLP(
        target_network.param_shapes,
        uncond_in_size=0,
        cond_in_size=parameters["embedding_size"],
        activation_fn=parameters["activation_function"],
        layers=parameters["hypernetwork_hidden_layers"],
        num_cond_embs=parameters["number_of_tasks"],
    ).to(parameters["device"])

    criterion = nn.CrossEntropyLoss()
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


def main_running_experiments(path_to_datasets, parameters):
    if parameters["dataset"] == "PermutedMNIST":
        dataset_tasks_list = prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"],
        )
    elif parameters["dataset"] == "SplitMNIST":
        dataset_tasks_list = prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            number_of_tasks=parameters["number_of_tasks"],
        )
    elif parameters["dataset"] == "CIFAR100":
        dataset_tasks_list = prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
        )
    elif parameters["dataset"] == "TinyImageNet":
        dataset_tasks_list = prepare_tinyimagenet_tasks(
            path_to_datasets,
            seed=parameters["seed"],
            validation_size=parameters["no_of_validation_samples"],
            number_of_tasks=parameters["number_of_tasks"]
        )
    elif parameters["dataset"] == "RotatedMNIST":
        dataset_tasks_list = prepare_rotated_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"],
        )
    elif parameters["dataset"] == "ImageNetSubset":
        dataset_tasks_list = prepare_imagenet_subset_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            input_shape=parameters["input_shape"],
        )

    # Measure time of the experiment
    start_time = time.time()

    hypernetwork, target_network, dataframe = build_multiple_task_experiment(
        dataset_tasks_list, parameters
    )

    elapsed_time = time.time() - start_time

    no_of_last_task = parameters["number_of_tasks"] - 1
    accuracies = dataframe.loc[dataframe["after_learning_of_task"] == no_of_last_task]["accuracy"].values
    row_with_results = (
        f"{dataset_tasks_list[0].get_identifier()};"
        f'{parameters["augmentation"]};'
        f'{parameters["embedding_size"]};'
        f'{parameters["seed"]};'
        f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
        f'{parameters["target_network"]};'
        f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
        f'{parameters["best_model_selection_method"]};'
        f'{parameters["optimizer"]};'
        f'{parameters["activation_function"]};'
        f'{parameters["learning_rate"]};{parameters["batch_size"]};'
        f'{parameters["perturbation_epsilon"]};'
        f"{np.mean(accuracies)};{np.std(accuracies)};"
        f"{elapsed_time}"
    )
    append_row_to_file(
        f'{parameters["grid_search_folder"]}'
        f'{parameters["summary_results_filename"]}.csv',
        row_with_results,
    )

    load_path = f'{parameters["saving_folder"]}/results_{parameters["dataset"]}.csv'
    plot_heatmap(load_path)

    return hypernetwork, target_network, dataframe