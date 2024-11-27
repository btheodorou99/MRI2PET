#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
#SBATCH --job-name=styleTransfer
#SBATCH --array=1-10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1

module load python
python -m src.utils.styleTransfer