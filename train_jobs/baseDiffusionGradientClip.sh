#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseDiffusionGradientClip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=baseDiffusionGradientClip.err
#SBATCH --output=baseDiffusionGradientClip.out

module load python
python -m src.train_scripts.train_baseDiffusionGradientClip