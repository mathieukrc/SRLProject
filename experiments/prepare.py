"""
Stage 0.

    # on the LOGIN node (needs internet for the DSRL download; light CPU work only):
    python -m experiments.prepare data --envs controlled point_goal car_circle

    # as a (small) array job: variants, round-0 critics and dynamics ensembles that the
    # manifest needs, each trained exactly once and shared by every targeting rule
    python -m experiments.prepare models --manifest runs/manifests/controlled.jsonl \\
        --shard $SLURM_ARRAY_TASK_ID --num-shards 4 --workers 4
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

from .config import ExperimentConfig


# --------------------------------------------------------------------- data
def check_h(spec, raw, tol: float = 0.01) -> bool:
    """sign(h(s')) must agree with the dataset's recorded cost; the analytic h
    otherwise defines a different constraint than the benchmark."""
    import torch

    nxt = torch.as_tensor(raw["next_observations"], dtype=torch.float32)
    h = torch.cat([spec.h_fn(nxt[i : i + 65536]) for i in range(0, len(nxt), 65536)])
    pred_unsafe = (h > 0).numpy()
    true_unsafe = np.asarray(raw["costs"]) > 0
    mismatch = float((pred_unsafe != true_unsafe).mean())
    print(
        f"  h check {spec.name}: mismatch {mismatch:.4%} "
        f"(dataset unsafe {true_unsafe.mean():.4%}, h unsafe {pred_unsafe.mean():.4%})"
    )
    return mismatch <= tol


def prepare_data(envs, strict: bool) -> int:
    from .envs import ENVS

    ok = True
    for name in envs:
        spec = ENVS[name]
        t0 = time.time()
        print(f"[{name}] loading (downloads / computes ground truth on first use)")
        raw = spec.load_raw()
        print(f"  {len(raw['observations'])} transitions, obs_dim={raw['observations'].shape[1]} ({time.time() - t0:.0f}s)")
        if spec.has_ground_truth:
            from controlled.env_api import check_feasible

            check_feasible(raw["observations"][:10])  # forces ground-truth computation
        ok &= check_h(spec, raw)
    if not ok:
        print("WARNING: h does not match the recorded costs for at least one env")
        return 1 if strict else 0
    return 0


# --------------------------------------------------------------------- models
def _prepare_one(args):
    cfg_dict, device, threads = args
    import torch

    torch.set_num_threads(threads)
    from . import pipeline
    from .envs import ENVS
    from .run import resolve_device

    cfg = ExperimentConfig.from_dict(cfg_dict).normalised()
    device = resolve_device(device)
    log = lambda m: print(f"[{cfg.round0_key}] {m}", flush=True)  # noqa: E731
    try:
        spec = ENVS[cfg.env]
        full = pipeline.load_full_dataset(spec, cfg, device)
        data = full.subset(pipeline.variant_indices(spec, full, cfg))
        dims = (data.data["observations"].shape[1], data.data["actions"].shape[1])
        pipeline.round0_critic(spec, cfg, full, data, dims, device, log)
        if cfg.dynamics == "learned" and cfg.rule != "none":
            pipeline.get_step_model(spec, cfg, data, device, log)
        return "ok"
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        return "failed"


def prepare_models(manifest, shard, num_shards, workers, threads, device) -> int:
    from .run import read_manifest

    cfgs = [ExperimentConfig.from_dict(c).normalised() for c in read_manifest(manifest)]
    # One representative per round-0 key; prefer one that needs dynamics so the
    # dynamics ensemble for that variant gets trained in the same pass.
    reps = {}
    for c in cfgs:
        cur = reps.get(c.round0_key)
        needs_dyn = c.dynamics == "learned" and c.rule != "none"
        if cur is None or (needs_dyn and not (cur.dynamics == "learned" and cur.rule != "none")):
            reps[c.round0_key] = c
    todo = [reps[k] for k in sorted(reps)][shard::num_shards]
    print(f"shard {shard}/{num_shards}: {len(todo)} round-0 critics", flush=True)

    jobs = [(c.to_dict(), device, threads) for c in todo]
    if workers <= 1:
        res = [_prepare_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            res = pool.map(_prepare_one, jobs, chunksize=1)
    print(f"ok={res.count('ok')} failed={res.count('failed')}", flush=True)
    return 1 if "failed" in res else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--envs", nargs="+", default=["controlled"])
    d.add_argument("--strict", action="store_true")
    m = sub.add_parser("models")
    m.add_argument("--manifest", required=True)
    m.add_argument("--shard", type=int, default=None)
    m.add_argument("--num-shards", type=int, default=1)
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--threads", type=int, default=1)
    m.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    if a.cmd == "data":
        return prepare_data(a.envs, a.strict)
    shard = a.shard if a.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    return prepare_models(a.manifest, shard, a.num_shards, a.workers, a.threads, a.device)


if __name__ == "__main__":
    sys.exit(main())
