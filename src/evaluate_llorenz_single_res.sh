#!/bin/bash
#SBATCH --job-name=evaluate_llorenz_single_res
#SBATCH --output=/scratch/work/%u/run_outputs/evaluate_llorenz_single_res_out_%A_%a.txt
#SBATCH --error=/scratch/work/%u/run_outputs/evaluate_llorenz_single_res_err_%A_%a.txt
#SBATCH --time=01:00:00
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=4G

python3 ./notebooks/evaluate_llorenz_single_res.py