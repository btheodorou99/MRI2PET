#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=12
#SBATCH --job-name=scalePretrain
#SBATCH --array=0-25

module load python
python -m src.utils.scalePretrain