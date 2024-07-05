#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bitsPerDimension
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=evaluateBPD.err
#SBATCH --output=evaluateBPD.out

module load python
python -m src.evaluation.calcBitsPerDim