# Sourced by every sbatch script. Configured for SURF Snellius.
set -euo pipefail
cd "${SRL_ROOT:?set SRL_ROOT to the repo checkout}"
module purge 2>/dev/null || true          # the venv brings its own Python 3.8
source .venv/bin/activate
export SRL_CACHE="${SRL_CACHE:-$HOME/srl_cache}"       # home: backed up, not purged
export SRL_RESULTS="${SRL_RESULTS:-$HOME/srl_results}"
export PYTHONPATH="$SRL_ROOT:${PYTHONPATH:-}"
mkdir -p logs

# Run WORKERS copies of a command in parallel inside one allocation, sharing its
# cores. Snellius bills CPU jobs per 16 cores and GPU jobs per GPU (+ its cores),
# so one process per job would pay for idle cores.
#   run_workers N "<command with {IDX} placeholder>"
run_workers() {
  local n=$1 cmd=$2 pids=() status=0
  local threads=$(( ${SLURM_CPUS_PER_TASK:-$n} / n ))
  [ "$threads" -ge 1 ] || threads=1
  for k in $(seq 0 $((n - 1))); do
    local idx=$(( ${SLURM_ARRAY_TASK_ID:-0} * n + k ))
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
      bash -c "${cmd//\{IDX\}/$idx}" > "logs/${SLURM_JOB_NAME:-job}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${idx}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  return $status
}
