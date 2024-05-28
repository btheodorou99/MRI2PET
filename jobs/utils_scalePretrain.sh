#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=scalePretrain
#SBATCH --array=0-50

module load python
python -m src.utils.scalePretrain