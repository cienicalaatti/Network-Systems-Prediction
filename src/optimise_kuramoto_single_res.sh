#!/bin/bash
#SBATCH --job-name=kuramoto_opt_single_res
#SBATCH --output=/scratch/work/%u/run_outputs/array_job_out_%A.txt
#SBATCH --error=/scratch/work/%u/run_outputs/array_job_err_%A.txt
#SBATCH --open-mode=append
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3000


ARGUMENTS1=$(sed '/##/d' ./argfiles/opt_kuramoto_args.txt)

python3 ./notebooks/optimise_hyperparameters_v3.py $ARGUMENTS1 --outfile=./outputs/kuramoto_opt/kuramoto_30_opt_single_res_4


