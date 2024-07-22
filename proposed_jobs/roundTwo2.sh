#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=roundTwo2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=roundTwo2.err
#SBATCH --output=roundTwo2.out

module load python
python -m src.train_scripts.train_roundTwo2