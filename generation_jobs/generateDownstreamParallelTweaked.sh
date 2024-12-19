#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=generateDownstreamParallelTweaked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --error=generateDownstreamParallelTweaked%A_%a.err
#SBATCH --output=generateDownstreamParallelTweaked%A_%a.err

module load python
python -m src.generation.generateDownstreamParallelTweaked