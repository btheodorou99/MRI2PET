#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=styleTransfer
#SBATCH --array=0-25

module load python
python -m src.utils.styleTransfer