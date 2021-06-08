#!/bin/bash
#SBATCH --time = 1:00:00
#SBATCH --job-name=mmd_experiment
#SBATCH --partition=amd-shortq
#SBATCH --output=/home/rgudmundarson/projects/MMDGraph/outputs/name=mmd_experiment-%j.out
#SBATCH --error=/home/rgudmundarson/projects/MMDGraph/errors/name=mmd_experiment-%j.err
#SBATCH --mail-user=rlg2000@hw.ac.uk
#SBATCH --mail-type=ALL

module purge
RUNPATH=/home/rgudmundarson/projects/MMDGraph
cd $RUNPATH
source .venv/bin/activate

python3 GraphMMDExperiments/BinomialGraphs.py -B 100 -N 100 -p 'data/binomial/ClassTest08062021.pkl' -s 1 -n 1
