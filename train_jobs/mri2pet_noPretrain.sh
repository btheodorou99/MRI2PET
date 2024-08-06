#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mri2pet_noPretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=mri2pet_noPretrain.err
#SBATCH --output=mri2pet_noPretrain.out

module load python
python -m src.train_scripts.train_MRI2PET_noPretrain