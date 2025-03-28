#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baselineFID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --error=baselineFID.err
#SBATCH --output=baselineFID.out

module load python
echo "FID"
python -m src.baselines.evaluation.calcFID