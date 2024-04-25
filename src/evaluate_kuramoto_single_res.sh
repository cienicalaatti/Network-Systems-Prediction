#!/bin/bash
#SBATCH --job-name=evaluate_kuramoto_single_res
#SBATCH --output=/scratch/work/%u/run_outputs/evaluate_kuramoto_single_res_%A.txt
#SBATCH --error=/scratch/work/%u/run_outputs/evaluate_kuramoto_single_res_%A.txt
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=10G

python3 ./notebooks/evaluate_kuramoto_single_res.py