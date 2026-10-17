"""
Shared artefacts, computed once and reused by every run.

Stage A, `--stage data`: controlled ground truth + dataset, DSRL downloads,
    h sign checks, fixed evaluation sets.
Stage B, `--stage baseline` (array over manifests/.../baselines_*.jsonl): per
    baseline_key: variant indices, dynamics ensemble on that variant, round-0
    critic. Its metrics are the premise test.

Usage:
  python -m experiments.prepare --stage data --envs controlled car_circle point_goal
  python -m experiments.prepare --stage baseline --manifest manifests/x/baselines_cpu.jsonl --index $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import List

import numpy as np
import torch

from data.offline_dataset import OfflineDataset

from .config import ExperimentConfig
from .grid import read_manifest
from .paths import (baseline_done_marker, baseline_path, cache_dir,
                    dynamics_path, variant_path)


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, using CPU")
        return "cpu"
    return device


def _write_json(path, obj) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)
    os.replace(tmp, str(path))


def load_full_dataset(cfg: ExperimentConfig, device: str) -> OfflineDataset:
    from envs.registry import get_env_spec, make_training_h

    spec = get_env_spec(cfg.env)
    training_h = make_training_h(spec, cfg.dense_h, cfg.h_pos_scale, cfg.h_neg_scale)
    return OfflineDataset(
        spec.load_raw_dataset(),
        device=device,
        n_members=cfg.ensemble_size,
        seed=cfg.seed,
        timeout_is_done=spec.timeout_is_done,
        h_fn=training_h,
    )


# --- stage A -----------------------------------------------------------------
def check_h_signs(env: str, raw: dict, tolerance: float = 0.005) -> float:
    """Fraction of transitions where sign(h(s')) disagrees with the recorded cost."""
    from envs.registry import get_env_spec

    spec = get_env_spec(env)
    with torch.no_grad():
        h = spec.h_fn(torch.as_tensor(raw["next_observations"], dtype=torch.float32)).numpy()
    mismatch = float(np.mean((h > 0) != (np.asarray(raw["costs"]) > 0)))
    msg = f"{env}: sign(h(s')) disagrees with the recorded cost on {mismatch:.4%} of transitions"
    if mismatch > tolerance:
        raise RuntimeError(msg + f" (> {tolerance:.2%}); fix cost_area.py before running")
    print(("[ok] " if mismatch == 0 else "[warn] ") + msg)
    return mismatch


def prepare_env_data(env: str) -> None:
    from envs.registry import get_env_spec
    from eval.evaluate import evaluation_states

    spec = get_env_spec(env)
    t = time.time()
    raw = spec.load_raw_dataset()  # controlled: computes ground truth; DSRL: downloads
    print(f"{env}: {len(raw['observations'])} transitions, obs_dim {raw['observations'].shape[1]} "
          f"({time.time() - t:.0f}s)")
    check_h_signs(env, raw)
    full = load_full_dataset(ExperimentConfig(env=env), "cpu")
    sets = evaluation_states(spec, full)
    print(f"{env}: evaluation sets " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))


# --- stage B -----------------------------------------------------------------
def build_variant_indices(cfg: ExperimentConfig, full: OfflineDataset) -> np.ndarray:
    from data.variant import get_coverage_variants, get_size_matched_variant
    from envs.registry import get_env_spec

    criterion = get_env_spec(cfg.env).boundary_criterion
    cov = get_coverage_variants(full, cfg.coverage_percent, criterion, seed=cfg.seed)
    if cfg.variant == "coverage":
        return cov.source_index
    return get_size_matched_variant(full, len(cov), seed=cfg.seed).source_index


def load_variant(cfg: ExperimentConfig, device: str) -> OfflineDataset:
    path = variant_path(cfg)
    full = load_full_dataset(cfg, device)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: baseline {cfg.baseline_key} not prepared")
    return full.subset(np.load(path)["index"])


def prepare_baseline(cfg: ExperimentConfig, device: str) -> None:
    from data.mixed import MixedSampler, SyntheticBuffer
    from data.variant import coverage_stats
    from dynamics.ensemble import (dynamics_validation_report, save_dynamics,
                                   train_dynamics_ensemble)
    from envs.registry import get_env_spec, make_training_h
    from eval.evaluate import (evaluate_feasibility, evaluation_states,
                               uncertainty_calibration)
    from FISOR.ensemble_feasibility_critic import EnsembleFeasibilityCritic
    from train.loops import train_feasibility_critic

    if baseline_done_marker(cfg).exists():
        return
    from data.utils import set_seed

    set_seed(cfg.seed)
    device = resolve_device(device)
    spec = get_env_spec(cfg.env)
    training_h = make_training_h(spec, cfg.dense_h, cfg.h_pos_scale, cfg.h_neg_scale)
    report = {"baseline_key": cfg.baseline_key}
    t0 = time.time()

    # 1. variant
    full = load_full_dataset(cfg, device)
    index = build_variant_indices(cfg, full)
    variant_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(variant_path(cfg), index=index)
    data = full.subset(index)
    report["coverage"] = coverage_stats(data, spec.boundary_criterion, reference=full)
    report["coverage_full"] = coverage_stats(full, spec.boundary_criterion)
    print(f"[{cfg.baseline_key}] variant: {report['coverage']}")
    del full

    # 2. dynamics ensemble on the VARIANT only
    perm = np.random.default_rng(cfg.seed).permutation(len(data))
    n_val = max(1, len(data) // 10)
    dyn = train_dynamics_ensemble(data.subset(perm[n_val:]), data.subset(perm[:n_val]),
                                  n_members=cfg.ensemble_size, seed=cfg.seed, verbose=False)
    save_dynamics(dyn, str(dynamics_path(cfg)))
    report["dynamics"] = dynamics_validation_report(dyn, data.subset(perm[:n_val]))
    print(f"[{cfg.baseline_key}] dynamics: {report['dynamics']}")

    # 3. round-0 critic
    obs_dim, act_dim = data.data["observations"].shape[1], data.data["actions"].shape[1]
    critic = EnsembleFeasibilityCritic(obs_dim, act_dim, K=cfg.ensemble_size,
                                       discount=cfg.discount, expectile=cfg.expectile,
                                       device=device)
    sampler = MixedSampler(data, SyntheticBuffer(obs_dim, act_dim, cfg.ensemble_size, device),
                           0.0, seed=cfg.seed)

    def log(info, step):
        print(f"[{cfg.baseline_key}] critic step {step}: vc={info['vc']:.3f} "
              f"loss={info['safe_value_loss']:.4f}", flush=True)

    ck = str(baseline_path(cfg)) + ".partial"
    critic = train_feasibility_critic(critic, sampler, cfg.base_steps,
                                      log_every=max(1, cfg.base_steps // 10), log_fn=log,
                                      checkpoint_path=ck, checkpoint_every=10_000)
    baseline_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(baseline_path(cfg)) + ".tmp"
    torch.save(critic.state_dict(), tmp)
    os.replace(tmp, str(baseline_path(cfg)))

    # 4. premise-test metrics
    eval_states = evaluation_states(spec)
    report["feasibility"] = evaluate_feasibility(critic, spec, eval_states, training_h)
    if spec.has_ground_truth:
        from controlled.env_api import check_feasible

        s = eval_states["random"]
        report["calibration"] = uncertainty_calibration(critic, s, check_feasible(s))
    report["seconds"] = time.time() - t0
    _write_json(cache_dir() / "baselines" / f"{cfg.baseline_key}.json", report)
    baseline_done_marker(cfg).touch()
    print(f"[{cfg.baseline_key}] done in {report['seconds']:.0f}s")


# --- CLI ---------------------------------------------------------------------
def _select(manifest: str, index: int, per_task: int) -> List[ExperimentConfig]:
    configs = read_manifest(manifest)
    return configs[index * per_task : (index + 1) * per_task]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["data", "baseline"], required=True)
    p.add_argument("--envs", nargs="*", default=["controlled", "car_circle", "point_goal"])
    p.add_argument("--manifest")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--per-task", type=int, default=1)
    p.add_argument("--all", action="store_true", help="every entry of the manifest")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    if args.stage == "data":
        for env in args.envs:
            prepare_env_data(env)
        return

    configs = read_manifest(args.manifest) if args.all else _select(args.manifest, args.index, args.per_task)
    failed = []
    for cfg in configs:
        try:
            prepare_baseline(cfg, args.device)
        except Exception:
            failed.append(cfg.baseline_key)
            print(f"[FAILED] {cfg.baseline_key}\n{traceback.format_exc()}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
