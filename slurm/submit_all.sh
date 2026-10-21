#!/bin/bash
# Submit every experiment at once, with the stages chained by job dependencies.
# Run from the repository root on a Snellius login node:
#
#     bash slurm/submit_all.sh smoke            # tiny runs through every code path (do this first)
#     bash slurm/submit_all.sh full             # the real thing
#     DRY_RUN=1 bash slurm/submit_all.sh full   # only print what would be submitted
#     GRIDS="controlled" bash slurm/submit_all.sh full   # a subset of the grids
#
# Budgets and partitions differ per account; check yours with `accinfo`.
#     NO_CPU_BUDGET=1 ...                  GPU budget only: the CPU grids run on GPU nodes,
#                                          and you aggregate on the login node afterwards
#     CPU_PARTITION=genoa CPU_CORES=24 ... a CPU budget on another partition
#     GPU_PARTITION=gpu_h100 ...           a GPU budget on another partition
#     ACCOUNT=<name from accinfo> ...      if you have more than one account
#
# Per grid:  prepare (round-0 critics, dynamics)  --afterany-->  run array
# Then:      aggregate  --afterany-->  all run arrays
# afterany (not afterok): if a prepare shard fails, the runs still start and train the
# missing round-0 critic themselves (guarded by a file lock), instead of never starting.
set -euo pipefail
MODE=${1:-smoke}
[[ "$MODE" == smoke || "$MODE" == full ]] || { echo "usage: $0 smoke|full"; exit 1; }
cd "$(dirname "$0")/.."
source slurm/env.sh

GRIDS=${GRIDS:-"controlled controlled_policy dsrl"}
SMOKE_FLAG=""; [[ "$MODE" == smoke ]] && SMOKE_FLAG="--smoke"

# ---- partitions and resources ------------------------------------------------
CPU_PARTITION=${CPU_PARTITION:-rome}
CPU_CORES=${CPU_CORES:-16}
GPU_PARTITION=${GPU_PARTITION:-gpu_a100}
GPU_RES="--partition=$GPU_PARTITION --gpus=1 --cpus-per-task=18"

if [[ -n "${NO_CPU_BUDGET:-}" ]]; then
    # CPU-type grids on one GPU: 8 workers that each use the GPU, rather than many slow
    # single-threaded CPU processes.
    CPU_RES="$GPU_RES"; CPU_DEVICE=cuda; CPU_WORKERS=8; CPU_PREP_WORKERS=4
else
    CPU_RES="--partition=$CPU_PARTITION --cpus-per-task=$CPU_CORES"
    CPU_DEVICE=cpu; CPU_WORKERS=$CPU_CORES; CPU_PREP_WORKERS=$CPU_CORES
fi
if [[ -n "${ACCOUNT:-}" ]]; then
    CPU_RES="$CPU_RES --account=$ACCOUNT"; GPU_RES="$GPU_RES --account=$ACCOUNT"
fi

# kind | runs per array task | workers per task | max concurrent tasks
declare -A KIND=( [controlled]=cpu [controlled_policy]=gpu [dsrl]=gpu )
declare -A COUNT=( [controlled]=32 [controlled_policy]=4 [dsrl]=4 )
declare -A WORKERS=( [controlled]=$CPU_WORKERS [controlled_policy]=4 [dsrl]=4 )
declare -A THROTTLE=( [controlled]=20 [controlled_policy]=4 [dsrl]=6 )

if [[ "$MODE" == smoke ]]; then
    TIME_RUN="--time=00:45:00"; TIME_PREP="--time=00:30:00"; TIME_AGG="--time=00:15:00"
else
    TIME_RUN=""; TIME_PREP=""; TIME_AGG=""   # use the defaults in the .job files
fi

submit() {  # prints the job id (or a fake one in dry-run mode)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

RUN_IDS=()
for G in $GRIDS; do
    NAME="$G"; [[ "$MODE" == smoke ]] && NAME="${G}_smoke"
    MANIFEST="$SRL_OUT/manifests/$NAME.jsonl"
    python -m experiments.grid "configs/$G.yaml" $SMOKE_FLAG >/dev/null
    N_RUNS=$(grep -c . "$MANIFEST")
    N_R0=$(python - "$MANIFEST" <<'PY'
import json, sys
from experiments.config import ExperimentConfig
print(len({ExperimentConfig.from_dict(json.loads(l)).round0_key for l in open(sys.argv[1]) if l.strip()}))
PY
)
    C=${COUNT[$G]}; W=${WORKERS[$G]}
    N_TASKS=$(( (N_RUNS + C - 1) / C ))
    OUT="--output=$SRL_OUT/slurm/%x_%A_%a.out"

    if [[ "${KIND[$G]}" == cpu ]]; then
        PREP_SHARDS=$(( (N_R0 + CPU_PREP_WORKERS - 1) / CPU_PREP_WORKERS ))
        PREP=$(submit --job-name="prep-$NAME" $OUT $TIME_PREP $CPU_RES \
            --array=0-$((PREP_SHARDS - 1)) \
            slurm/prepare_models.job "$MANIFEST" "$PREP_SHARDS" "$CPU_PREP_WORKERS" "$CPU_DEVICE")
        RUN=$(submit --job-name="run-$NAME" $OUT $TIME_RUN $CPU_RES \
            --dependency=afterany:$PREP --array=0-$((N_TASKS - 1))%${THROTTLE[$G]} \
            slurm/run_cpu.job "$MANIFEST" "$C" "$W" "$CPU_DEVICE")
    else
        PREP_SHARDS=$(( (N_R0 + 3) / 4 ))
        PREP=$(submit --job-name="prep-$NAME" $OUT $TIME_PREP $GPU_RES \
            --array=0-$((PREP_SHARDS - 1)) \
            slurm/prepare_models.job "$MANIFEST" "$PREP_SHARDS" 4 cuda)
        RUN=$(submit --job-name="run-$NAME" $OUT $TIME_RUN $GPU_RES \
            --dependency=afterany:$PREP --array=0-$((N_TASKS - 1))%${THROTTLE[$G]} \
            slurm/run_gpu.job "$MANIFEST" "$C" "$W")
    fi
    RUN_IDS+=("$RUN")
    printf "%-24s %4d runs, %3d round-0 critics -> prep %-10s (%d tasks), run %-10s (%d tasks, %s)\n" \
        "$NAME" "$N_RUNS" "$N_R0" "$PREP" "$PREP_SHARDS" "$RUN" "$N_TASKS" "${KIND[$G]}"
done

if [[ -n "${NO_CPU_BUDGET:-}" ]]; then
    # Aggregation only reads JSON and draws a few plots: not worth GPU budget.
    echo "aggregate: when the jobs are done, on the login node run"
    echo "           source slurm/env.sh && python -m experiments.aggregate"
else
    DEPS=$(IFS=:; echo "${RUN_IDS[*]}")
    AGG=$(submit --job-name="agg-$MODE" --output="$SRL_OUT/slurm/%x_%j.out" $TIME_AGG $CPU_RES \
        --dependency=afterany:$DEPS slurm/aggregate.job)
    echo "aggregate: $AGG   (results: $SRL_OUT/summary)"
fi
echo "monitor:   squeue -u \$USER    |   python -m experiments.aggregate   (any time)"
