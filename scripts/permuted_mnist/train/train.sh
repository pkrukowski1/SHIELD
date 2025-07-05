#!/bin/bash
#SBATCH --job-name=shield_train_permuted_mnist
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/permuted_mnist/train/train"