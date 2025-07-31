#!/bin/bash
#SBATCH --job-name=shield_split_permuted_mnist_cil_autoattack
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090_batch

source scripts/main.sh

run_sweep_and_agent "scripts/permuted_mnist/cil_autoattack/cil_autoattack"
