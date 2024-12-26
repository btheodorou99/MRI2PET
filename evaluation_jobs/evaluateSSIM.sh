#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=structuralSimilarityIndex
#SBATCH --error=evaluateSSIM.err
#SBATCH --output=evaluateSSIM.out

module load python
python -m src.evaluation.calcSSIM