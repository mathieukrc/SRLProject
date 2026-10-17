#!/bin/bash
#SBATCH --job-name=srl-base-gpu
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# 1 A100 + 18 cores = 1/4 node, the smallest gpu_a100 allocation.
source slurm/common.sh
run_workers "${WORKERS_GPU:-2}" "python -m experiments.prepare --stage baseline --device cuda \
  --manifest $MANIFEST_DIR/baselines_gpu.jsonl --index {IDX}"
