#!/bin/bash
# Submit the whole pipeline with dependencies:  data -> baselines -> runs -> aggregate
#
#   SRL_ROOT=$PWD bash slurm/submit_all.sh configs/smoke.json                 # always first
#   SRL_ROOT=$PWD bash slurm/submit_all.sh configs/grid_controlled.json configs/grid_dsrl.json
#
# Env: SKIP_DATA=1 if stage A was already run (e.g. on the login node, when compute
# nodes have no internet for the DSRL download). THROTTLE caps concurrent array tasks.
set -euo pipefail
: "${SRL_ROOT:?set SRL_ROOT to the repo checkout}"
cd "$SRL_ROOT"
export SRL_ROOT
# runs per worker process, and worker processes per job (see slurm/common.sh)
export PER_TASK_CPU="${PER_TASK_CPU:-2}" PER_TASK_GPU="${PER_TASK_GPU:-1}"
export WORKERS_CPU="${WORKERS_CPU:-4}" WORKERS_GPU="${WORKERS_GPU:-2}"
NAME="${NAME:-$(basename "${1%.json}")}"
export MANIFEST_DIR="$SRL_ROOT/manifests/$NAME"
THROTTLE="${THROTTLE:-50}"
mkdir -p logs

source .venv/bin/activate
python -m experiments.grid --spec "$@" --out "$MANIFEST_DIR" \
  --per-task-cpu $((PER_TASK_CPU * WORKERS_CPU)) --per-task-gpu $((PER_TASK_GPU * WORKERS_GPU))

lines() { [ -f "$1" ] && grep -c . "$1" || echo 0; }
tasks() { echo $(( ( $1 + $2 - 1 ) / $2 )); }

submit_array() {   # script n_tasks dependency -> prints job id (or nothing if n=0)
  local script=$1 n=$2 dep=$3
  [ "$n" -gt 0 ] || return 0
  sbatch --parsable --export=ALL ${dep:+--dependency=$dep} \
    --array=0-$((n - 1))%"$THROTTLE" "$script"
}

dep_data=""
if [ -z "${SKIP_DATA:-}" ]; then
  A=$(sbatch --parsable --export=ALL slurm/prepare_data.sbatch)
  dep_data="afterok:$A"
fi

BC=$(submit_array slurm/baseline_cpu.sbatch "$(tasks "$(lines "$MANIFEST_DIR/baselines_cpu.jsonl")" "$WORKERS_CPU")" "$dep_data")
BG=$(submit_array slurm/baseline_gpu.sbatch "$(tasks "$(lines "$MANIFEST_DIR/baselines_gpu.jsonl")" "$WORKERS_GPU")" "$dep_data")

# Runs wait for ALL baseline arrays (a GPU run can use a CPU baseline, e.g. a
# controlled-env run with train_agent). afterany: one failed baseline should not
# cancel unrelated runs; runs whose baseline is missing fail fast with a clear message.
base_deps=$(echo "${BC:-} ${BG:-}" | xargs | tr ' ' ':')
RC=$(submit_array slurm/run_cpu.sbatch \
  "$(tasks "$(lines "$MANIFEST_DIR/runs_cpu.jsonl")" $((PER_TASK_CPU * WORKERS_CPU)))" "${base_deps:+afterany:$base_deps}")
RG=$(submit_array slurm/run_gpu.sbatch \
  "$(tasks "$(lines "$MANIFEST_DIR/runs_gpu.jsonl")" $((PER_TASK_GPU * WORKERS_GPU)))" "${base_deps:+afterany:$base_deps}")

deps=$(echo "$RC $RG" | xargs | tr ' ' ':')
AG=$(sbatch --parsable --export=ALL ${deps:+--dependency=afterany:$deps} slurm/aggregate.sbatch)

echo "data=${A:-skipped} base_cpu=${BC:-none} base_gpu=${BG:-none} runs_cpu=${RC:-none} runs_gpu=${RG:-none} aggregate=$AG"
echo "Re-running this script later only resubmits work: finished runs are skipped (DONE markers)."
