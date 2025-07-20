#!/bin/bash
#SBATCH --job-name=shield_train_split_mini_imagenet_autoattack
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

source scripts/main.sh

run_sweep_and_agent "scripts/split_mini_imagenet/autoattack/autoattack"