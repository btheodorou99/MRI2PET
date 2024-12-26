#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=paDiffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=paDiffusion.err
#SBATCH --output=paDiffusion.out

module load python
python -m src.baselines.train_scripts.train_paDiffusion
python -m src.baselines.generation.generate_paDiffusion