#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baselineNonFID
#SBATCH --error=baselineNonFID.err
#SBATCH --output=baselineNonFID.out

module load python
echo "PSNR"
python -m src.baselines.evaluation.calcPSNR
echo "SSIM"
python -m src.baselines.evaluation.calcSSIM