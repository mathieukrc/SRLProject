"""
Expand grid specs (JSON) into ExperimentConfigs and write manifests.

Spec format:
{
  "base":   {...ExperimentConfig fields shared by every block...},
  "blocks": [
    {"name": "main",
     "axes": {"coverage_percent": [100, 20], "rule": [...], "budget": [...], "seed": [0, 1]},
     "override": {...fixed fields for this block only...}},
    ...
  ]
}
An axis whose values are dicts is "zipped": each dict is merged as a whole, e.g.
  "components": [{"use_transitions": true, "use_anchors": false}, ...]
Rule "none" is collapsed to budget 0, and duplicates across blocks are removed.
Keys in "base" starting with "_" are ignored (use them as comments).

Usage:
  python -m experiments.grid --spec configs/grid_controlled.json configs/grid_dsrl.json --out manifests
Writes manifests/{runs,baselines}_{cpu,gpu}.jsonl and prints the array sizes.
"""

import argparse
import itertools
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List

from .config import ExperimentConfig


def load_grid_spec(path: str) -> Dict[str, Any]:
    with open(path) as f:
        spec = json.load(f)
    if "blocks" not in spec:
        raise ValueError(f"{path}: grid spec needs a 'blocks' list")
    return spec


def _expand_block(base: Dict[str, Any], block: Dict[str, Any]) -> List[ExperimentConfig]:
    axes = block.get("axes", {})
    names = list(axes)
    out = []
    for values in itertools.product(*(axes[n] for n in names)):
        d = {k: v for k, v in base.items() if not k.startswith("_")}
        d.update(block.get("override", {}))
        for name, value in zip(names, values):
            if isinstance(value, dict):
                d.update(value)  # zipped axis
            else:
                d[name] = value
        if d.get("rule", "none") == "none":
            d["budget"] = 0
        d["tag"] = block.get("name", "")
        out.append(ExperimentConfig.from_dict(d))
    return out


def make_grid(spec: Dict[str, Any]) -> List[ExperimentConfig]:
    """All runs in the spec, deduplicated by run_id (the first block keeps the tag)."""
    runs: "OrderedDict[str, ExperimentConfig]" = OrderedDict()
    for block in spec["blocks"]:
        for cfg in _expand_block(spec.get("base", {}), block):
            runs.setdefault(cfg.run_id, cfg)
    return list(runs.values())


def baseline_configs(configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
    """One config per distinct baseline_key (stage-B jobs)."""
    seen: "OrderedDict[str, ExperimentConfig]" = OrderedDict()
    for cfg in configs:
        seen.setdefault(cfg.baseline_key, cfg.baseline_config())
    return list(seen.values())


def split_by_resource(
    configs: List[ExperimentConfig],
) -> Dict[str, List[ExperimentConfig]]:
    out: Dict[str, List[ExperimentConfig]] = {"cpu": [], "gpu": []}
    for cfg in configs:
        out[cfg.resource].append(cfg)
    return out


def write_manifest(configs: List[ExperimentConfig], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for cfg in configs:
            f.write(json.dumps(cfg.to_dict(), sort_keys=True) + "\n")
    return len(configs)


def read_manifest(path: str) -> List[ExperimentConfig]:
    with open(path) as f:
        return [ExperimentConfig.from_dict(json.loads(l)) for l in f if l.strip()]


def n_array_tasks(n_runs: int, per_task: int) -> int:
    return (n_runs + per_task - 1) // per_task


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", nargs="+", required=True)
    p.add_argument("--out", default="manifests")
    p.add_argument("--per-task-cpu", type=int, default=8)
    p.add_argument("--per-task-gpu", type=int, default=1)
    args = p.parse_args(argv)

    configs: List[ExperimentConfig] = []
    for s in args.spec:
        configs += make_grid(load_grid_spec(s))
    configs = list(OrderedDict((c.run_id, c) for c in configs).values())

    runs = split_by_resource(configs)
    bases = split_by_resource(baseline_configs(configs))
    per = {"cpu": args.per_task_cpu, "gpu": args.per_task_gpu}

    for res in ("cpu", "gpu"):
        n = write_manifest(runs[res], os.path.join(args.out, f"runs_{res}.jsonl"))
        nb = write_manifest(bases[res], os.path.join(args.out, f"baselines_{res}.jsonl"))
        print(
            f"{res}: {n} runs -> {n_array_tasks(n, per[res])} array tasks "
            f"(x{per[res]} runs each), {nb} baselines"
        )

    counts: Dict[str, int] = {}
    for c in configs:
        counts[c.tag] = counts.get(c.tag, 0) + 1
    print("runs per block:", counts)


if __name__ == "__main__":
    main()
