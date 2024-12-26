#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dclGAN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=dclGAN.err
#SBATCH --output=dclGAN.out

module load python
python -m src.baselines.train_scripts.train_dclGAN
python -m src.baselines.generation.generate_dclGAN