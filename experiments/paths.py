"""
Directory layout. Everything goes under $SRL_OUT (default: <repo>/runs).

On Snellius, point SRL_OUT at project space or /scratch-shared/$USER (see slurm/env.sh).
Scratch is fast but is cleaned up automatically after a while, so copy the aggregated
results somewhere permanent when you are done.

    $SRL_OUT/
      cache/variants/<variant_key>.npy      row indices of each dataset variant
      cache/round0/<round0_key>.pt          round-0 critic, shared by every targeting rule
      cache/dynamics/<dynamics_key>.pt      dynamics ensembles (learned-dynamics runs)
      manifests/<grid>.jsonl                one config per line
      results/<run_id>/                     config.json, metrics.json, log.txt, DONE
      summary/                              aggregated CSVs and plots
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_root() -> str:
    return os.environ.get("SRL_OUT", os.path.join(REPO, "runs"))


def sub(*parts: str) -> str:
    p = os.path.join(out_root(), *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def variant_path(key: str) -> str:
    return sub("cache", "variants", key + ".npy")


def round0_path(key: str) -> str:
    return sub("cache", "round0", key + ".pt")


def dynamics_path(key: str) -> str:
    return sub("cache", "dynamics", key + ".pt")


def manifest_path(name: str) -> str:
    return sub("manifests", name + ".jsonl")


def result_dir(run_id: str) -> str:
    return sub("results", run_id)


def summary_dir() -> str:
    return sub("summary")
