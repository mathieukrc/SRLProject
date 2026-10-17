"""
Stage D: collect every results/<run_id>/ into tables.

Usage:
  python -m experiments.aggregate --manifests manifests/runs_cpu.jsonl manifests/runs_gpu.jsonl --out analysis
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import ExperimentConfig
from .grid import read_manifest
from .paths import done_marker, results_dir, run_dir

GROUP_COLS = (
    "tag", "env", "variant", "coverage_percent", "dense_h", "rule", "budget",
    "n_rounds", "use_transitions", "use_anchors", "transition_model",
)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
    return out


def collect_results(configs: List[ExperimentConfig]) -> pd.DataFrame:
    """One row per finished run: config columns + flattened final.json."""
    rows = []
    for cfg in configs:
        if not done_marker(cfg).exists():
            continue
        final = json.loads((run_dir(cfg) / "final.json").read_text())
        rows.append({**cfg.to_dict(), "run_id": cfg.run_id, **_flatten(final)})
    return pd.DataFrame(rows)


def collect_rounds(configs: List[ExperimentConfig]) -> pd.DataFrame:
    """One row per (run, round), for learning-over-rounds plots."""
    rows = []
    for cfg in configs:
        path = run_dir(cfg) / "rounds.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                rows.append({**cfg.to_dict(), "run_id": cfg.run_id, **json.loads(line)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metric_cols: Sequence[str], ci: float = 0.95) -> pd.DataFrame:
    """Mean, std, n and t-based CI half-width over seeds."""
    group = [c for c in GROUP_COLS if c in df.columns]
    g = df.groupby(group, dropna=False)[list(metric_cols)]
    mean, std, n = g.mean(), g.std(), g.count()
    tcrit = stats.t.ppf(0.5 + ci / 2, np.maximum(n - 1, 1))
    half = pd.DataFrame(tcrit * std.values / np.sqrt(np.maximum(n.values, 1)),
                        index=mean.index, columns=mean.columns)
    return pd.concat({"mean": mean, "std": std, "ci": half, "n": n}, axis=1)


def paired_differences(
    df: pd.DataFrame, metric_cols: Sequence[str], reference_rule: str = "density_matched"
) -> pd.DataFrame:
    """
    rule minus reference_rule at the same seed and setting. Runs with the same
    seed share the baseline critic, so pairing removes most seed variance.
    """
    keys = [c for c in GROUP_COLS if c in df.columns and c not in ("rule",)] + ["seed"]
    ref = df[df["rule"] == reference_rule].set_index(keys)[list(metric_cols)]
    out = []
    for rule, sub in df[df["rule"] != reference_rule].groupby("rule"):
        diff = sub.set_index(keys)[list(metric_cols)] - ref
        diff = diff.dropna(how="all").reset_index()
        diff["rule"] = rule
        out.append(diff)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    return summarise(d, metric_cols)


def report_missing(configs: List[ExperimentConfig]) -> List[str]:
    """run_ids without DONE (failed or never ran); FAILED ones first."""
    missing = [c for c in configs if not done_marker(c).exists()]
    failed = [c.run_id for c in missing if (run_dir(c) / "FAILED").exists()]
    other = [c.run_id for c in missing if c.run_id not in failed]
    return failed + other


def _lineplot(df, x, y, hue, ax, title):
    for name, sub in df.groupby(hue):
        g = sub.groupby(x)[y]
        m, se = g.mean(), g.std() / np.sqrt(g.count().clip(lower=1))
        ax.errorbar(m.index, m.values, yerr=se.fillna(0).values, marker="o", capsize=3, label=str(name))
    ax.set_xlabel(x)
    ax.set_title(title, fontsize=9)


def plot_main_figures(df: pd.DataFrame, rounds: pd.DataFrame, out_dir: Path) -> None:
    """Quick-look figures; any metric that is missing is skipped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    main = df[df.get("tag", "") == "main"] if "tag" in df else df
    for env, sub in main.groupby("env"):
        metrics = [m for m in (
            "feasibility/random/false_feasible_rate", "feasibility/random/false_infeasible_rate",
            "feasibility/random/iou_feasible", "feasibility/band/accuracy",
            "feasibility/proxy/precursor_called_feasible", "feasibility/proxy/pred_feasible_frac",
            "policy/cost_mean", "policy/return_mean",
        ) if m in sub.columns and sub[m].notna().any()]
        for budget, sb in sub.groupby("budget"):
            if budget == 0 or not metrics:
                continue
            sb = pd.concat([sb, sub[sub["rule"] == "none"]])
            fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5), squeeze=False)
            for ax, m in zip(axes[0], metrics):
                _lineplot(sb, "coverage_percent", m, "rule", ax, m.split("/", 1)[-1])
            axes[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{env}_vs_coverage_b{budget}.png", dpi=120)
            plt.close(fig)

    if not rounds.empty and "random/false_feasible_rate" in rounds.columns:
        r = rounds[(rounds["tag"] == "main") & (rounds["env"] == "controlled")]
        for cov, sub in r.groupby("coverage_percent"):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            _lineplot(sub, "round", "random/false_feasible_rate", "rule", axes[0], f"false feasible, cov {cov}")
            _lineplot(sub, "round", "random/false_infeasible_rate", "rule", axes[1], f"false infeasible, cov {cov}")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"rounds_cov{int(cov)}.png", dpi=120)
            plt.close(fig)

    if not rounds.empty and "syn/trans_in_true_band" in rounds.columns:
        r = rounds[rounds["round"] > 0].groupby("rule")[["syn/trans_in_true_band", "syn/anchor_in_true_band"]].mean()
        ax = r.plot.bar(figsize=(6, 3.5), rot=30)
        ax.set_title("fraction of synthetic data in the true boundary band", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "targeting_band_fraction.png", dpi=120)
        plt.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--out", default="analysis")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    configs: List[ExperimentConfig] = []
    for m in args.manifests:
        configs += read_manifest(m)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing = report_missing(configs)
    (out / "missing.txt").write_text("\n".join(missing))
    print(f"{len(configs) - len(missing)}/{len(configs)} runs done; "
          f"{len(missing)} missing (see {out / 'missing.txt'}); results in {results_dir()}")

    df = collect_results(configs)
    rounds = collect_rounds(configs)
    df.to_csv(out / "runs.csv", index=False)
    rounds.to_csv(out / "rounds.csv", index=False)
    if df.empty:
        return

    metrics = [c for c in df.columns if c.startswith(("feasibility/", "policy/", "calibration/"))]
    summarise(df, metrics).to_csv(out / "summary.csv")
    paired_differences(df, metrics).to_csv(out / "paired_vs_density_matched.csv")
    if not args.no_plots:
        plot_main_figures(df, rounds, out)


if __name__ == "__main__":
    main()
