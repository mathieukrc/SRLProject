#!/bin/bash
#SBATCH --job-name=srl-data
#SBATCH --partition=staging
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# staging = SURF's partition for data transfer; it downloads the DSRL datasets.
# If the download fails here, run stage A on a login node instead (see SNELLIUS.md).
source slurm/common.sh
python -m experiments.prepare --stage data --envs ${ENVS:-controlled car_circle point_goal}
