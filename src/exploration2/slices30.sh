#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=slices30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=slices30.err
#SBATCH --output=slices30.out

module load python
python -m src.exploration2.slices30