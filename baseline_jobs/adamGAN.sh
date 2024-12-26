#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=adamGAN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=adamGAN.err
#SBATCH --output=adamGAN.out

module load python
python -m src.baselines.train_scripts.train_adamGAN
python -m src.baselines.generation.generate_adamGAN