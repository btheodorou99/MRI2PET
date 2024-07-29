#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=generateDataset8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=generateDataset8.err
#SBATCH --output=generateDataset8.out

module load python
python -m src.generation.generateDataset8