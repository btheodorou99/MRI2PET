#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
#SBATCH --job-name=styleTransfer
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1,lscratch:100

export TMPDIR=/lscratch/$SLURM_JOB_ID
export OMP_NUM_THREADS=1

module load python
python -m src.utils.styleTransfer