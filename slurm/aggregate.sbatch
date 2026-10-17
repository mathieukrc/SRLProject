#!/bin/bash
#SBATCH --job-name=srl-agg
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
source slurm/common.sh
python -m experiments.aggregate --out "$SRL_ROOT/analysis/$(basename "$MANIFEST_DIR")" \
  --manifests "$MANIFEST_DIR/runs_cpu.jsonl" "$MANIFEST_DIR/runs_gpu.jsonl"
