#!/bin/bash
#SBATCH --job-name=srl-base-cpu
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# 16 cores = the smallest (and cheapest) rome allocation; WORKERS_CPU baselines share it.
source slurm/common.sh
run_workers "${WORKERS_CPU:-4}" "python -m experiments.prepare --stage baseline --device cpu \
  --manifest $MANIFEST_DIR/baselines_cpu.jsonl --index {IDX}"
