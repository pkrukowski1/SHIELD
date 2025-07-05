#!/bin/bash
#SBATCH --job-name=shield_train_tiny_imagenet
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/tiny_imagenet/train/train"