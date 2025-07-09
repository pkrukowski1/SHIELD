#!/bin/bash
#SBATCH --job-name=shield_interval_mixup_permuted_mnist_eps_scheduling
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

source scripts/main.sh

run_sweep_and_agent "scripts/permuted_mnist/interval_mixup_eps_scheduling/interval_mixup_eps_scheduling"