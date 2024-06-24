#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=petPostprocess
#SBATCH --array=0-25

module load python
python -m src.utils.PET_PostProcessing