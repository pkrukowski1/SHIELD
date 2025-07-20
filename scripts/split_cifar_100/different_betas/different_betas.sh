#!/bin/bash
#SBATCH --job-name=shield_interval_mixup_split_cifar_100_different_betas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

source scripts/main.sh

run_sweep_and_agent "scripts/split_cifar_100/different_betas/different_betas"