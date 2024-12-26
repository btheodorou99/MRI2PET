#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=peak_signal_to_noise_ratio
#SBATCH --error=evaluatePSNR.err
#SBATCH --output=evaluatePSNR.out

module load python
python -m src.evaluation.calcPSNR