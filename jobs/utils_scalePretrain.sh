#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=247g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=scalePretrain

module load python
python -m src.utils.scalePretrain