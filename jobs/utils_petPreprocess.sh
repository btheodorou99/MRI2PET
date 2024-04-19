#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=24
#SBATCH --job-name=petPreprocess

module load python
python -m src.utils.PET_PreProcessing