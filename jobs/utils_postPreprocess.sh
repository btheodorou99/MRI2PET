#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=247g
#SBATCH --cpus-per-task=24
#SBATCH --job-name=petPostprocess

module load python
python -m ../src/utils/PET_PostProcessing.py