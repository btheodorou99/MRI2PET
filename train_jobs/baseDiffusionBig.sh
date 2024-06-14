#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseDiffusionBig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=baseDiffusionBig.err
#SBATCH --output=baseDiffusionBig.out

module load python
python -m src.train_scripts.train_baseDiffusionBig