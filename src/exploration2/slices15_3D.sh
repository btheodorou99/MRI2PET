#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=slices15_3D
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=slices15_3D.err
#SBATCH --output=slices15_3D.out

module load python
python -m src.exploration2.slices15_3D