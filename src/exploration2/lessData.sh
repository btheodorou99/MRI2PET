#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=lessData
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=lessData.err
#SBATCH --output=lessData.out

module load python
python -m src.exploration2.lessData