#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tweak_baseDiffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=tweak_baseDiffusion.err
#SBATCH --output=tweak_baseDiffusion.out

module load python
python -m src.train_scripts.tweak_baseDiffusion