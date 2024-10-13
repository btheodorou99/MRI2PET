#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mri2pet_base_loss2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=tune_mri2pet_base_loss2.err
#SBATCH --output=tune_mri2pet_base_loss2.out

module load python
python -m src.train_scripts.tune_MRI2PET_base_loss2