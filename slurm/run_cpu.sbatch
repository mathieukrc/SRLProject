#!/bin/bash
#SBATCH --job-name=srl-run-cpu
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
source slurm/common.sh
run_workers "${WORKERS_CPU:-4}" "python -m experiments.run --device cpu --per-task $PER_TASK_CPU \
  --manifest $MANIFEST_DIR/runs_cpu.jsonl --index {IDX}"
