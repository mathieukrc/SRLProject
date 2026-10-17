#!/bin/bash
#SBATCH --job-name=srl-run-gpu
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# WORKERS_GPU runs share one A100: a single FISOR run does not saturate it.
source slurm/common.sh
run_workers "${WORKERS_GPU:-2}" "python -m experiments.run --device cuda --per-task $PER_TASK_GPU \
  --manifest $MANIFEST_DIR/runs_gpu.jsonl --index {IDX}"
