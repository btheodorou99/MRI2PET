#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=noisy_base_loss
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=tune_noisy_base_loss.err
#SBATCH --output=tune_noisy_base_loss.out

module load python
python -m src.train_scripts.tune_noisy_base_loss