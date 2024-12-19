#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=structuralSimilarityIndex
#SBATCH --error=evaluateSSIM.err
#SBATCH --output=evaluateSSIM.out

module load python
echo "Average SSIM"
python -m src.evaluation.calcSSIM_average
echo "Average Content SSIM"
python -m src.evaluation.calcSSIM_average_content
echo "Channel Axis SSIM"
python -m src.evaluation.calcSSIM_channel_axis
echo "Multichannel SSIM"
python -m src.evaluation.calcSSIM_multichannel
echo "Multichannel Content SSIM"
python -m src.evaluation.calcSSIM_multichannel_content