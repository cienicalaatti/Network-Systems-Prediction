# Reservoir Computing in Predicting Networks of Coupled Dynamical Systems

Repository containing the source code to produce the analysis of the thesis work. The parallel reservoir scheme implementation is based on a [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.164101) by Keshav et. al., and a modified version of [ReservoirPy](https://reservoirpy.readthedocs.io/en/latest/) (v0.2) Python package. The analysis was run on a Slurm-based computing cluster.

This repository contains the essential source code for the most important parts of the analysis, not all data/scripts required to reproduce all the results presented in the thesis are included.

### To run the optimise the model and run a prediction:
(0. set up a Python virtual environment according to [requirements.txt](src/requirements.txt))
1. Simulate data (notebooks in [src/notebooks_dynamic_systems](src/notebooks_dynamic_systems))
2. Run model optimisation:
   1. set optimisation parameters ([src/argfiles](src/argfiles))
   2. run optimisation in a slurm environment (optimisation bash script in [src](src))
3. Run prediction:
   1. set prediction parameters ([src/argfiles](src/argfiles))
   2. run prediction in a slurm environment (shared_memory_parallel_run bash script in [src](src))
4. Visualise results (notebooks in [src/notebooks_analysis](src/notebooks_analysis))
