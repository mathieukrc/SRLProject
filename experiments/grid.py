"""
Expand a grid file into a manifest (one JSON config per line).

    python -m experiments.grid configs/controlled.yaml            # full grid
    python -m experiments.grid configs/controlled.yaml --smoke    # tiny version, same code paths

Grid file format (YAML):

    name: controlled
    base: {env: controlled, rounds: 3}          # applies to every block
    blocks:
      - axes: {coverage: [100, 10], rule: [none, uncertainty_guided], seed: [0, 1]}
      - base: {rule: uncertainty_guided}         # block-specific overrides
        axes: {label_noise: [0.05, 0.1], seed: [0, 1]}
    smoke:
      overrides: {round0_steps: 200, steps_per_round: 100}
      axes: {seed: [0]}                          # replaces these axes in every block

Configs are normalised and deduplicated by run id (e.g. rule "none" ignores the
budget, size_matched at 100% equals coverage at 100%).
"""

import argparse
import itertools
import json
from typing import Dict, List

from . import paths
from .config import ExperimentConfig


def load_grid(path: str) -> dict:
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        import yaml

        return yaml.safe_load(f)


def expand(grid: dict, smoke: bool = False) -> List[dict]:
    base = dict(grid.get("base", {}))
    smoke_cfg = grid.get("smoke", {}) if smoke else {}
    seen, out = set(), []
    for block in grid["blocks"]:
        b_base = dict(base, **block.get("base", {}))
        axes: Dict[str, list] = dict(block.get("axes", {}))
        for k, v in smoke_cfg.get("axes", {}).items():
            if k in axes:
                axes[k] = v
        keys = list(axes)
        for values in itertools.product(*(axes[k] for k in keys)):
            d = dict(b_base, **dict(zip(keys, values)))
            d.update(smoke_cfg.get("overrides", {}))
            cfg = ExperimentConfig.from_dict(d).normalised()
            if cfg.run_id in seen:
                continue
            seen.add(cfg.run_id)
            out.append(cfg.to_dict())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("grid")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--name", default=None, help="manifest name (default: grid name)")
    p.add_argument("--count", type=int, default=None, help="runs per array task, to print the array size")
    a = p.parse_args(argv)

    grid = load_grid(a.grid)
    cfgs = expand(grid, a.smoke)
    name = a.name or (grid["name"] + ("_smoke" if a.smoke else ""))
    path = paths.manifest_path(name)
    with open(path, "w") as f:
        for c in cfgs:
            f.write(json.dumps(c) + "\n")

    n_round0 = len({ExperimentConfig.from_dict(c).round0_key for c in cfgs})
    print(f"{len(cfgs)} runs ({n_round0} distinct round-0 critics) -> {path}")
    if a.count:
        n_tasks = (len(cfgs) + a.count - 1) // a.count
        print(f"array: --array=0-{n_tasks - 1} with --count {a.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
