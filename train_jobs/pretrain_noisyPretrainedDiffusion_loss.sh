#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=noisyPretrainedDiffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=pretrain_noisyPretrainedDiffusion_loss.err
#SBATCH --output=pretrain_noisyPretrainedDiffusion_loss.out

module load python
python -m src.train_scripts.pretrain_noisyPretrainedDiffusion_loss