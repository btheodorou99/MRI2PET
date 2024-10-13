#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mri2pet_noPretrain_noNoise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=mri2pet_noPretrain_noNoise.err
#SBATCH --output=mri2pet_noPretrain_noNoise.out

module load python
python -m src.train_scripts.train_MRI2PET_noPretrain_noNoise