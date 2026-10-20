#!/bin/bash
# Sourced by every job script and by submit_all.sh. Edit the ACTIVATE part to match
# how you created the environment in slurm/setup_env.sh.

# ---- locations ---------------------------------------------------------------
export SRL_REPO="${SRL_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# Results + caches. $HOME is backed up and has room (200 GB); /scratch-shared/$USER is
# faster but cleaned automatically, so copy results out if you use it.
export SRL_OUT="${SRL_OUT:-$HOME/srl_runs}"
# DSRL looks here for its .hdf5 files (downloaded once, on the login node).
export DSRL_DATASET_DIR="${DSRL_DATASET_DIR:-$HOME/.dsrl/datasets}"
export SRL_ENV_PREFIX="${SRL_ENV_PREFIX:-$HOME/envs/srl}"

# ---- software ----------------------------------------------------------------
module purge >/dev/null 2>&1 || true
# ACTIVATE: conda env created by setup_env.sh (Python 3.8, as DSRL requires)
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "$SRL_ENV_PREFIX"
else
    echo "env.sh: miniforge not found in \$HOME/miniforge3 - edit slurm/env.sh" >&2
fi

# ---- runtime -----------------------------------------------------------------
export MPLBACKEND=Agg          # no display on compute nodes
export PYTHONUNBUFFERED=1      # logs appear in the .out files immediately
export OMP_NUM_THREADS=1       # each worker is one process; parallelism comes from workers
export MKL_NUM_THREADS=1
export PYTHONPATH="$SRL_REPO:${PYTHONPATH:-}"

cd "$SRL_REPO"
mkdir -p "$SRL_OUT/slurm"
