# Running the experiments

## Layout

```
configs/            grid files: controlled.yaml, controlled_policy.yaml, dsrl.yaml
experiments/        config (run ids), envs (registry), pipeline, run, grid, prepare, aggregate
generation/         synthetic chains: action proposal, rollout + filter, budgeted generator
evaluation/         critic metrics (ground truth / DSRL proxy, calibration), policy rollouts
dynamics/ensemble.py  dynamics ensemble with calibrated disagreement threshold
slurm/              setup_env.sh, env.sh, *.job, submit_all.sh
```

One run = one config = one folder `$SRL_OUT/results/<run_id>/` with `config.json`,
`metrics.json`, `log.txt` and `DONE` (or `FAILED` with the traceback). Run ids are hashes
of the config, so resubmitting skips finished runs and resumes interrupted ones.

## Snellius, first time (login node, from the repo root)

```bash
bash slurm/setup_env.sh          # Miniforge + Python 3.8 env, torch, DSRL, datasets, h checks
```

Check the h-check lines at the end: the mismatch should be ~0% for every env. If not,
the analytic h in `cost_area.py` defines a different constraint than the benchmark.

## Every submission

```bash
bash slurm/submit_all.sh smoke             # minutes; exercises every code path
python -m experiments.aggregate            # look at $SRL_OUT/summary/status.csv
DRY_RUN=1 bash slurm/submit_all.sh full    # inspect the sbatch commands
bash slurm/submit_all.sh full
GRIDS="controlled" bash slurm/submit_all.sh full   # only some grids
```

Use the smoke `log.txt` timings to set `--time` in the `.job` files and to estimate SBUs.

## Local debugging (no Slurm)

```bash
export SRL_OUT=./runs
python -m experiments.prepare data --envs controlled
python -m experiments.grid configs/controlled.yaml --smoke
python -m experiments.run --manifest runs/manifests/controlled_smoke.jsonl --workers 2
python -m experiments.run --config '{"env": "controlled", "rule": "uncertainty_guided", "budget_frac": 0.05, "round0_steps": 500, "rounds": 1, "steps_per_round": 100}'
python -m experiments.aggregate
```

## Adding things

- New targeting rule: add a picker to `data/synthetic.py`, its name to
  `START_TARGETING_RULES`, and a branch in `generation/generator.py:pick_starts`.
- New task: one `EnvSpec` in `experiments/envs.py` (loader, h_fn, valid_fn,
  terminal_fn, boundary criterion, eval env).
- New experiment block: a `blocks:` entry in a grid file. Any `ExperimentConfig` field
  can be set in `base:` or swept in `axes:`.

## Fairness rules the pipeline enforces

- Every rule, including `none`, runs the same rounds and gradient steps.
- The round-0 critic is shared by all rules of a variant/seed (trained once, cached).
- The budget counts transitions *after* filtering.
- Synthetic data trains the feasibility critic only; the policy stage trains the actor
  and reward critic on real data with the feasibility critic frozen.
