#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=frechetInceptionDistance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --error=evaluateFID.err
#SBATCH --output=evaluateFID.out

module load python
python -m src.evaluation.calcFID