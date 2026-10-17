"""
All filesystem locations in one place. Everything is absolute and driven by
environment variables, so jobs behave the same whatever their working directory.

  SRL_CACHE    shared artefacts: ground truth, DSRL data, variants, dynamics, baselines
  SRL_RESULTS  one directory per run

NOTE: controlled/controlled_env.py currently uses OUT_DIR = "controlled/env_outputs"
(relative to the CWD). Point it at env_outputs_dir() instead, otherwise array jobs
started from another directory silently recompute the ground truth, and parallel
jobs can race on writing ground_truth.npz / dataset.npz.
"""

import os
from pathlib import Path

from .config import ExperimentConfig


def cache_dir() -> Path:
    return Path(os.environ.get("SRL_CACHE", "cache")).resolve()


def results_dir() -> Path:
    return Path(os.environ.get("SRL_RESULTS", "results")).resolve()


def env_outputs_dir() -> Path:
    """Replacement for controlled_env.OUT_DIR."""
    return cache_dir() / "controlled"


def variant_path(cfg: ExperimentConfig) -> Path:
    """Kept-transition indices (npz) for the run's dataset variant."""
    return cache_dir() / "variants" / f"{cfg.baseline_key}.npz"


def dynamics_path(cfg: ExperimentConfig) -> Path:
    """Dynamics ensemble trained ONLY on the run's dataset variant."""
    return cache_dir() / "dynamics" / f"{cfg.baseline_key}.pt"


def baseline_path(cfg: ExperimentConfig) -> Path:
    """Round-0 ensemble feasibility critic shared by all rules."""
    return cache_dir() / "baselines" / f"{cfg.baseline_key}.pt"


def baseline_done_marker(cfg: ExperimentConfig) -> Path:
    return cache_dir() / "baselines" / f"{cfg.baseline_key}.done"


def run_dir(cfg: ExperimentConfig) -> Path:
    return results_dir() / cfg.run_id


def done_marker(cfg: ExperimentConfig) -> Path:
    return run_dir(cfg) / "DONE"
