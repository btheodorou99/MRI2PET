#!/bin/bash
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=12
#SBATCH --job-name=petPreprocess
#SBATCH --array=1-50

module load python
python -m src.utils.PET_PreProcessingSafe