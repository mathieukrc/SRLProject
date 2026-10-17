# Experiment pipeline

Unzip at the repo root (overwrites the 9 modified files listed below; `git diff` shows the changes).

See SNELLIUS.md for running on Snellius.

Stages:  A data  ->  B baselines (premise test)  ->  C runs  ->  D aggregate

## On the cluster
    SRL_ROOT=$PWD bash slurm/submit_all.sh configs/smoke.json        # first, always
    SRL_ROOT=$PWD bash slurm/submit_all.sh configs/grid_controlled.json configs/grid_dsrl.json
Re-running submit_all.sh only redoes unfinished work (DONE markers; runs resume from
the last completed round). Edit slurm/common.sh (modules, partitions) first. If compute
nodes have no internet, run stage A on the login node and submit with SKIP_DATA=1.

## Locally (one process, no SLURM)
    export SRL_CACHE=$PWD/cache SRL_RESULTS=$PWD/results
    python -m experiments.grid --spec configs/smoke.json --out manifests/smoke
    python -m experiments.prepare --stage data --envs controlled
    python -m experiments.prepare --stage baseline --all --manifest manifests/smoke/baselines_cpu.jsonl --device cpu
    python -m experiments.run --manifest manifests/smoke/runs_cpu.jsonl --per-task 999 --device cpu
    python -m experiments.aggregate --manifests manifests/smoke/runs_cpu.jsonl manifests/smoke/runs_gpu.jsonl --out analysis/smoke
Debug one run: python -m experiments.run --manifest ... --run-id <run_id> --device cpu

## Outputs
    cache/baselines/<key>.json   premise test: coverage stats, dynamics quality, round-0 metrics
    results/<run_id>/            config.json, rounds.jsonl, final.json, map.npz, DONE | FAILED
    analysis/<name>/             runs.csv, rounds.csv, summary.csv, paired_vs_density_matched.csv,
                                 missing.txt, *.png

## Modified existing files
    controlled/controlled_env.py    OUT_DIR follows $SRL_CACHE
    data/offline_dataset.py         subset() tracks source_index, no deep copy of tensors
    data/utils.py                   label_costs() shared by real and synthetic labels
    data/synthetic.py               candidate_fn for the adaptive pickers
    FISOR/ensemble_feasibility_critic.py, feasibility_critic.py   anchor term in update()
    FISOR/reward_critic.py          ignores bootstrap_weights / is_synthetic keys
    FISOR/agent.py                  injected critic; synthetic rows kept out of the reward
                                    critic (and actor unless use_synthetic_actions)
    test/test_synthetic.py          dummy_violation returns float scores

## Semantics worth knowing
    budget            total ACCEPTED synthetic transitions over all rounds (split evenly)
    n_anchors         0 -> same as budget
    max_disagreement  multiple of the 95th-percentile ensemble disagreement on real
                      validation data (stored in the dynamics checkpoint)
    rule "none"       same extra training steps as every other rule, no synthetic data
