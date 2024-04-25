#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=parallel_opt_all
#SBATCH --output=/scratch/work/%u/run_outputs/parallel_opt_all_out_%A.txt
#SBATCH --error=/scratch/work/%u/run_outputs/parallel_opt_all_err_%A.txt

ARGUMENTS1=$(sed '/##/d' ./argfiles/parallel_opt_kuramoto_args.txt)
python3 ./notebooks/optimise_hyperparameters_parallel_all.py $ARGUMENTS1