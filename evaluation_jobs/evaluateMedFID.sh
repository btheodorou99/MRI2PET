#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=medFrechetInceptionDistance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --error=evaluateMedFID.err
#SBATCH --output=evaluateMedFID.out

module load python
python -m src.evaluation.calcMedFID