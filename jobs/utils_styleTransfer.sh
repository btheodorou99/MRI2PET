#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=24g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=styleTransfer
#SBATCH --array=0-25

module load python
python -m src.utils.styleTransfer