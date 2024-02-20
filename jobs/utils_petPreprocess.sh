#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=512g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=petPreprocess

module load python
python -m ../src/utils/PET_PreProcessing.py