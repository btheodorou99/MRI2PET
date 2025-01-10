#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baselineEval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --error=baselineEval.err
#SBATCH --output=baselineEval.out

module load python
echo "FID"
python -m src.baselines.evaluation.calcFID
echo "IS"
python -m src.baselines.evaluation.calcIS
echo "PSNR"
python -m src.baselines.evaluation.calcPSNR
echo "SSIM"
python -m src.baselines.evaluation.calcSSIM