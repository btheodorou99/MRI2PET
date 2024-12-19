#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=structuralSimilarityIndex
#SBATCH --error=evaluatePSNR.err
#SBATCH --output=evaluatePSNR.out

module load python
echo "Average PSNR"
python -m src.evaluation.calcPSNR_average
echo "Average Content PSNR"
python -m src.evaluation.calcPSNR_multichannel
echo "Multichannel Content PSNR"
python -m src.evaluation.calcPSNR_multichannel_content