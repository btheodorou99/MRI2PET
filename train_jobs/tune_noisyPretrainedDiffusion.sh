#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tune_noisyPretrainedDiffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=tune_noisyPretrainedDiffusion.err
#SBATCH --output=tune_noisyPretrainedDiffusion.out

module load python
python -m src.train_scripts.tune_noisyPretrainedDiffusion