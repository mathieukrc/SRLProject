"""
Stage 2: collect every finished run into tidy CSVs and a few standard plots.

    python -m experiments.aggregate              # all results under $SRL_OUT
    python -m experiments.aggregate --tag main   # only runs with this tag

Outputs in $SRL_OUT/summary/:
    long.csv      one row per (run, round, eval_set, metric)
    summary.csv   mean / std / n / 95% CI over seeds
    status.csv    done / failed / incomplete per run (check this first)
    plots/*.png
"""

import argparse
import glob
import json
import os

import numpy as np

from . import paths

GROUP = ["env", "variant", "coverage", "rule", "budget_frac", "label_noise", "dynamics", "round", "eval_set", "metric"]
# Config fields that never define a condition: averaged over (seed) or irrelevant.
NOT_A_CONDITION = {"seed", "tag", "log_every"}


def collect(tag=None):
    import pandas as pd

    rows, status = [], []
    for d in sorted(glob.glob(os.path.join(paths.out_root(), "results", "*"))):
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            meta = json.load(f)
        cfg = meta["config"]
        if tag is not None and cfg.get("tag") != tag:
            continue
        st = "done" if os.path.exists(os.path.join(d, "DONE")) else (
            "failed" if os.path.exists(os.path.join(d, "FAILED")) else "incomplete")
        status.append({"run_id": meta["run_id"], "status": st, "git": meta.get("git"), **cfg})
        if st != "done":
            continue
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        flat = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in cfg.items()}
        for r in m["records"]:
            rows.append({"run_id": meta["run_id"], "git": meta.get("git"), **flat, **r})
    return pd.DataFrame(rows), pd.DataFrame(status)


def condition_columns(long):
    """GROUP plus every other config field that takes more than one value among the
    runs (e.g. discount in a discount sweep), so different settings are never averaged
    together. Only seeds are averaged over."""
    from experiments.config import ExperimentConfig
    import dataclasses

    fields = [f.name for f in dataclasses.fields(ExperimentConfig)]
    extra = [k for k in fields
             if k in long.columns and k not in GROUP and k not in NOT_A_CONDITION
             and long[k].astype(str).nunique() > 1]
    return GROUP[:-3] + extra + GROUP[-3:], extra


def summarise(long):
    cols, extra = condition_columns(long)
    if extra:
        print("also grouping by:", ", ".join(extra))
    g = long.groupby(cols, dropna=False)["value"]
    s = g.agg(["mean", "std", "count"]).reset_index()
    s["ci95"] = 1.96 * s["std"].fillna(0) / np.sqrt(s["count"].clip(lower=1))
    return s


def plot(summary, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    last = summary[summary["round"] == summary.groupby("env")["round"].transform("max")]
    metrics = {
        "controlled": [("random", "iou_feasible"), ("random", "false_feasible_rate"),
                       ("random", "false_infeasible_rate"), ("band", "accuracy")],
    }
    for env, df in last.groupby("env"):
        wanted = metrics.get(env, [("real_proxy", "proxy_auroc"), ("policy", "normalized_cost"), ("policy", "normalized_return")])
        for (budget, variant), sub in df.groupby(["budget_frac", "variant"]):
            fig, axes = plt.subplots(1, len(wanted), figsize=(4.2 * len(wanted), 3.6), squeeze=False)
            for ax, (es, metric) in zip(axes[0], wanted):
                sel = sub[(sub.eval_set == es) & (sub.metric == metric)]
                # the no-augmentation baseline has budget 0: draw it on every plot
                base = df[(df.rule == "none") & (df.eval_set == es) & (df.metric == metric) & (df.variant == variant)]
                for rule, r in list(sel.groupby("rule")) + ([("none", base)] if budget > 0 else []):
                    r = r.sort_values("coverage")
                    ax.errorbar(r["coverage"], r["mean"], yerr=r["ci95"], marker="o", capsize=3, label=rule)
                ax.set_title(f"{es}: {metric}", fontsize=9)
                ax.set_xlabel("boundary coverage (%)")
            axes[0][0].legend(fontsize=7)
            fig.suptitle(f"{env} | {variant} | budget {budget}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{env}_{variant}_budget{budget}.png"), dpi=130)
            plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)

    long, status = collect(a.tag)
    out = paths.summary_dir()
    status.to_csv(os.path.join(out, "status.csv"), index=False)
    if len(status):
        print(status["status"].value_counts().to_string())
    if long.empty:
        print("no finished runs yet")
        return 0
    cols, _ = condition_columns(long)
    keep = ["run_id", "seed", "git"] + cols + [c for c in ("value",) if c in long]
    long[keep].to_csv(os.path.join(out, "long.csv"), index=False)
    s = summarise(long)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    plot(s, os.path.join(out, "plots"))
    print(f"{long['run_id'].nunique()} runs aggregated -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
