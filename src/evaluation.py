from method.interval_mixup_decay_rate import MixupEpsilonDecayRate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_decay_rates(save_path: str) -> None:
    """
    Plots different epsilon decay rate functions used in Interval MixUp and saves the figure.

    Args:
        save_path (str): The file path where the plot image will be saved (e.g., 'decay_rates.png').

    The function compares the following decay schedules:
        - Linear
        - Logarithmic
        - Quadratic
        - Cosine
    """
    # Initialize decay functions
    linear_decay_fnc = MixupEpsilonDecayRate("linear")
    log_decay_fnc = MixupEpsilonDecayRate("log")
    quadratic_decay_fnc = MixupEpsilonDecayRate("quadratic")
    cos_decay_fnc = MixupEpsilonDecayRate("cos")

    # Generate alpha values and compute decay outputs
    alpha = np.linspace(0, 1.0, 100)
    linear_decay_out = np.array([linear_decay_fnc(a) for a in alpha])
    log_decay_out = np.array([log_decay_fnc(a) for a in alpha])
    quadratic_decay_out = np.array([quadratic_decay_fnc(a) for a in alpha])
    cos_decay_out = np.array([cos_decay_fnc(a) for a in alpha])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(alpha, linear_decay_out, label="Linear")
    plt.plot(alpha, log_decay_out, label="Logarithmic")
    plt.plot(alpha, quadratic_decay_out, label="Quadratic")
    plt.plot(alpha, cos_decay_out, label="Cosine")

    plt.title("Epsilon Decay Rate Functions in Interval MixUp", fontsize=16)
    plt.xlabel("Alpha", fontsize=14)
    plt.ylabel("Epsilon Scale", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path)
    plt.close()


def plot_acc_diff_decay_rates(
        folder_path: str,
        linear_decay_path: str,
        quadratic_decay_path: str,
        log_decay_path: str,
        cos_decay_path: str
) -> None:
    """
    Plots accuracy trends for different epsilon decay rates in Interval MixUp:
    (1) Accuracy after learning each task.
    (2) Accuracy on all tasks after learning the final task.

    Args:
      folder_path (str): The folder path where the plot images will be saved.
      linear_decay_path, quadratic_decay_path, log_decay_path, cos_decay_path (str):
      Paths to CSV files containing accuracy logs.
    """
    def load_data(path: str, label: str):
        df = pd.read_csv(path, sep=';')
        max_task = df['after_learning_of_task'].max()

        # Accuracy immediately after learning each task (diagonal)
        diag = df[df['after_learning_of_task'] == df['tested_task']].sort_values('tested_task')['accuracy'].values

        # Accuracy on each task after learning all tasks (final row block)
        final = df[df['after_learning_of_task'] == max_task].sort_values('tested_task')['accuracy'].values

        return diag, final, label

    # Load and process all decay data
    results = [
        load_data(linear_decay_path, "Linear"),
        load_data(quadratic_decay_path, "Quadratic"),
        load_data(log_decay_path, "Logarithmic"),
        load_data(cos_decay_path, "Cosine"),
    ]

    num_tasks = len(results[0][0])  # assumes all runs have same number of tasks
    x = list(range(1, num_tasks + 1))

    # Plot 1: Accuracy after learning each task
    plt.figure(figsize=(8, 5))
    for diag, _, label in results:
        plt.plot(x, diag, marker='o', label=label)
    plt.title("Accuracy After Learning Each Task", fontsize=14)
    plt.xlabel("Task Index", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder_path}/accuracy_after_each_task.png")
    plt.close()

    # Plot 2: Accuracy on all tasks after learning final task
    plt.figure(figsize=(8, 5))
    for _, final, label in results:
        plt.plot(x, final, marker='o', label=label)
    plt.title("Accuracy on All Tasks After Learning Final Task", fontsize=14)
    plt.xlabel("Task Index", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.grid(True)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder_path}/accuracy_after_final_task.png")
    plt.close()
    

if __name__ == "__main__":
    plot_decay_rates("./ablation_study/interval_mixup/decay_rates.png")

    # PermutedMNIST
    plot_acc_diff_decay_rates(
        folder_path="./ablation_study/interval_mixup/permuted_mnist",
        linear_decay_path="./saved_models/permuted_mnist/mixup/linear/results.csv",
        quadratic_decay_path="./saved_models/permuted_mnist/mixup/quadratic/results.csv",
        log_decay_path="./saved_models/permuted_mnist/mixup/log/results.csv",
        cos_decay_path="./saved_models/permuted_mnist/mixup/cos/results.csv"
    )

    # Split-CIFAR100
    plot_acc_diff_decay_rates(
        folder_path="./ablation_study/interval_mixup/split_cifar_100",
        linear_decay_path="./saved_models/split_cifar_100/mixup/linear/results.csv",
        quadratic_decay_path="./saved_models/split_cifar_100/mixup/quadratic/results.csv",
        log_decay_path="./saved_models/split_cifar_100/mixup/log/results.csv",
        cos_decay_path="./saved_models/split_cifar_100/mixup/cos/results.csv"
    )
