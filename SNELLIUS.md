# Running on Snellius

## 1. One-time setup (login node: `ssh <user>@snellius.surf.nl`)

    git clone https://github.com/mathieukrc/SRLProject.git && cd SRLProject
    unzip -o ~/srl_pipeline.zip          # after copying it over with scp/rsync

    # Python 3.8 (needed by DSRL; not in the 2023/2024 module stacks) via uv
    curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
    uv venv --python 3.8 .venv && source .venv/bin/activate
    uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    git clone https://github.com/liuzuxin/DSRL.git && uv pip install -e DSRL
    uv pip install -r requirements.txt

    # download the DSRL datasets now (light work, fine on a login node)
    python -c "from data.utils import load_all_dsrl; load_all_dsrl()"

## 2. Check the environment on a real GPU node (a few minutes, a few SBU)

    srun -p gpu_a100 --gpus=1 --cpus-per-task=18 -t 00:20:00 --pty bash
    source .venv/bin/activate
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    python -c "import dsrl, gymnasium as gym; e = gym.make('OfflineCarCircle-v0'); e.reset(); print(e.step(e.action_space.sample())[-1])"
    exit
If the second command fails on safety-gymnasium / MuJoCo, `uv pip install safety-gymnasium`
and retry. The printed dict must contain a "cost" key (eval/evaluate.py reads info["cost"]).

## 3. Smoke test, then the real grid

    SRL_ROOT=$PWD bash slurm/submit_all.sh configs/smoke.json
    squeue -u $USER                      # watch it
    cat analysis/smoke/missing.txt       # must be empty when it finishes
    SRL_ROOT=$PWD bash slurm/submit_all.sh configs/grid_controlled.json configs/grid_dsrl.json

If stage A fails to download on the staging partition, it was already done in step 1:
resubmit with `SKIP_DATA=1` after running `python -m experiments.prepare --stage data`
inside an `srun -p rome -c 16 -t 01:00:00 --pty bash` session.

Re-running the same submit command only redoes unfinished or failed runs.

## Where things go
    $HOME/srl_cache     ground truth, DSRL eval sets, variants, dynamics, baselines
    $HOME/srl_results   one directory per run
    logs/               SLURM logs; one log per worker: <job>_<arrayjob>_<index>.log
    analysis/<grid>/    tables and plots
Home is backed up and not purged; all of this is a few GB.

## Packing (why the scripts look like this)
Snellius bills CPU jobs in blocks of 16 cores and GPU jobs per GPU plus its cores
(gpu_a100: 1 GPU = 1/4 node = 18 cores). One small process per job would pay for idle
cores, so each job runs several worker processes (slurm/common.sh: run_workers):
    WORKERS_CPU=4  x PER_TASK_CPU=2 runs per rome job (16 cores, 4 threads each)
    WORKERS_GPU=2  x PER_TASK_GPU=1 runs per A100 job
Override on the command line, e.g. `WORKERS_GPU=3 SRL_ROOT=$PWD bash slurm/submit_all.sh ...`.

## Budget
Check what you have and what you used with `accinfo`. Measure the smoke runs, then
estimate: SBU ≈ job-hours x 16 (rome) or x 128 (one A100) — confirm the rates on the
SURF "Snellius partitions and accounting" page.
