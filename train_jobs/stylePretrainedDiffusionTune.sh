#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=stylePretrainedDiffusionTune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=stylePretrainedDiffusionTune.err
#SBATCH --output=stylePretrainedDiffusionTune.out

module load python
python -m src.train_scripts.train_stylePretrainedDiffusionTune