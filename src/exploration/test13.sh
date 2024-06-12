#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test13
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1

module load python
python -m src.exploration.train_baseDiffusionTest13