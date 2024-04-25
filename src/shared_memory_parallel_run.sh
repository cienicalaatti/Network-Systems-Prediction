#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=sm_parallel_run
#SBATCH --output=/scratch/work/%u/run_outputs/sm_parallel_run_out_%A.txt
#SBATCH --error=/scratch/work/%u/run_outputs/sm_parallel_run_err_%A.txt

python3 ./notebooks/shared_memory_parallel_run.py