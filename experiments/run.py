"""
Run experiments from a manifest.

    # one array task = a chunk of `--count` manifest lines, run by `--workers` processes
    python -m experiments.run --manifest runs/manifests/controlled.jsonl \\
        --chunk $SLURM_ARRAY_TASK_ID --count 32 --workers 16

    # a single config, e.g. for debugging on a login node / laptop
    python -m experiments.run --config '{"env": "controlled", "rule": "none", "round0_steps": 500}'

Every run is idempotent: a finished run (DONE marker) is skipped, an interrupted run
resumes from its last completed round, and a failed run leaves FAILED with a traceback.
Resubmitting the same array therefore only redoes what is missing.
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import List, Optional

from . import paths
from .config import ExperimentConfig


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_one(cfg_dict: dict, device: str = "auto", threads: int = 1, force: bool = False) -> str:
    """Runs one config. Returns 'done', 'skipped' or 'failed'. Never raises."""
    import torch

    torch.set_num_threads(threads)
    cfg = ExperimentConfig.from_dict(cfg_dict).normalised()
    out = paths.result_dir(cfg.run_id)
    done, failed = os.path.join(out, "DONE"), os.path.join(out, "FAILED")
    if os.path.exists(done) and not force:
        return "skipped"
    if os.path.exists(failed):
        os.remove(failed)

    log_file = open(os.path.join(out, "log.txt"), "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_file.write(line + "\n")
        log_file.flush()

    device = resolve_device(device)
    meta = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "git": git_sha(),
        "host": socket.gethostname(),
        "device": device,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)

    t0 = time.time()
    log(f"start {cfg.run_id} on {meta['host']} ({device})")
    try:
        from .pipeline import atomic_json, run_experiment

        records = run_experiment(cfg, out, device, log)
        atomic_json({"run_id": cfg.run_id, "wall_time_s": time.time() - t0, "records": records},
                    os.path.join(out, "metrics.json"))
        open(done, "w").close()
        state = os.path.join(out, "state.pt")
        if os.path.exists(state):  # only needed for resuming; can be large
            os.remove(state)
        log(f"done in {time.time() - t0:.0f}s")
        return "done"
    except Exception:
        tb = traceback.format_exc()
        log("FAILED\n" + tb)
        with open(failed, "w") as f:
            f.write(tb)
        return "failed"
    finally:
        log_file.close()


def _worker(args):
    cfg_dict, device, threads, force = args
    status = run_one(cfg_dict, device, threads, force)
    rid = ExperimentConfig.from_dict(cfg_dict).normalised().run_id
    print(f"{status:8s} {rid}", flush=True)
    return status


def read_manifest(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--config", help="a single config as JSON (debugging)")
    p.add_argument("--chunk", type=int, default=None, help="array task id; uses SLURM_ARRAY_TASK_ID if unset")
    p.add_argument("--count", type=int, default=None, help="manifest lines per chunk (default: all)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="rerun finished runs")
    a = p.parse_args(argv)

    if a.config:
        cfgs = [json.loads(a.config)]
    elif a.manifest:
        cfgs = read_manifest(a.manifest)
        if a.count is not None:
            chunk = a.chunk if a.chunk is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            cfgs = cfgs[chunk * a.count : (chunk + 1) * a.count]
    else:
        p.error("give --manifest or --config")

    print(f"{len(cfgs)} runs, {a.workers} workers x {a.threads} threads, results in {paths.out_root()}", flush=True)
    jobs = [(c, a.device, a.threads, a.force) for c in cfgs]
    if a.workers <= 1:
        statuses = [_worker(j) for j in jobs]
    else:
        # spawn: CUDA cannot be used in forked children
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            statuses = pool.map(_worker, jobs, chunksize=1)

    n_failed = statuses.count("failed")
    print(f"done={statuses.count('done')} skipped={statuses.count('skipped')} failed={n_failed}", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
