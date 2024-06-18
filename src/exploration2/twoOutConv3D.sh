#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=twoOutConv3D
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=twoOutConv3D.err
#SBATCH --output=twoOutConv3D.out

module load python
python -m src.exploration2.twoOutConv3D