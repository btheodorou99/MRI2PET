#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=noisyPretrainedDiffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=noisyPretrainedDiffusion.err
#SBATCH --output=noisyPretrainedDiffusion.out

module load python
python -m src.train_scripts.train_noisyPretrainedDiffusion