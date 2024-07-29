#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=generateDataset5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=generateDataset5.err
#SBATCH --output=generateDataset5.out

module load python
python -m src.generation.generateDataset5