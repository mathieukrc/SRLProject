#!/bin/bash
# One-time setup, run on a Snellius LOGIN node from the repository root:
#     bash slurm/setup_env.sh
# Login nodes have internet; compute nodes may not, so everything that downloads
# (packages, DSRL datasets) happens here.
set -euo pipefail
ENV_PREFIX="${SRL_ENV_PREFIX:-$HOME/envs/srl}"

# 1. Miniforge (user-space conda) - Snellius' module stacks do not ship Python 3.8.
if [ ! -d "$HOME/miniforge3" ]; then
    curl -L -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 2. Python 3.8 environment
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.8
conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip

# 3. PyTorch: 2.4.x is the last release with Python 3.8 wheels; the cu121 build runs
#    on the A100/H100 nodes and also works on CPU-only nodes.
python -m pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121

# 4. DSRL from source (the pip release has the dataset-path issue noted in README.md)
[ -d DSRL ] || git clone https://github.com/liuzuxin/DSRL.git
python -m pip install -e DSRL
python -m pip install -r requirements.txt

# 5. Datasets + controlled-env ground truth + h-function checks (light CPU work)
source slurm/env.sh
python -m experiments.prepare data --envs controlled point_goal car_circle

echo "setup done. Next: bash slurm/submit_all.sh smoke"
