#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=doubleConv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=doubleConv.err
#SBATCH --output=doubleConv.out

module load python
python -m src.exploration2.doubleConv