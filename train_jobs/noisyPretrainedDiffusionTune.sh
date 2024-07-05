#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=noisyPretrainedDiffusionTune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=noisyPretrainedDiffusionTune.err
#SBATCH --output=noisyPretrainedDiffusionTune.out

module load python
python -m src.train_scripts.train_noisyPretrainedDiffusionTune