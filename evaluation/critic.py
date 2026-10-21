"""
Critic-level evaluation.

controlled env : exact metrics against the grid ground truth (test/metrics.py), plus
                 calibration of the ensemble's p_feasible.
DSRL tasks     : no ground truth. Proxy label: a real state is "empirically infeasible"
                 if its own trajectory violates the constraint within `horizon` steps.
                 This is only a proxy (the behaviour policy may have been careless from a
                 feasible state), so report it as such.
"""

from typing import Dict, List

import numpy as np
import torch

from data.offline_dataset import OfflineDataset
from data.variant import split_trajectories


@torch.no_grad()
def _batched(fn, states: np.ndarray, device, chunk: int = 65536) -> np.ndarray:
    out = []
    for i in range(0, len(states), chunk):
        x = torch.as_tensor(states[i : i + chunk], dtype=torch.float32, device=device)
        out.append(fn(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def calibration(p_feasible: np.ndarray, truly_feasible: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Expected calibration error of p_feasible as a probability of being feasible,
    plus the share of misclassified states in the most and least uncertain halves."""
    p = np.clip(np.asarray(p_feasible, dtype=np.float64), 0, 1)
    y = np.asarray(truly_feasible).astype(np.float64)
    if len(p) == 0:
        return {}
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())

    pred = p >= 0.5
    wrong = pred != y.astype(bool)
    # With K members p only takes K+1 values, so a median split is degenerate. Split on
    # whether the members agree instead: is the ensemble's disagreement informative?
    agree = (p == 0) | (p == 1)
    return {
        "ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "frac_members_disagree": float((~agree).mean()),
        "err_rate_members_agree": float(wrong[agree].mean()) if agree.any() else float("nan"),
        "err_rate_members_disagree": float(wrong[~agree].mean()) if (~agree).any() else float("nan"),
    }


def evaluate_controlled(critic, full_real: OfflineDataset) -> List[dict]:
    """One record per (eval_set, metric). full_real: the *original* full dataset, so all
    variants are scored on the same states."""
    from controlled.env_api import _ground_truth, check_feasible
    from test.metrics import evaluation_sets_controlled_env, feasibility_metrics

    sets = evaluation_sets_controlled_env(full_real)
    # Extra set: random safe states inside the true boundary band, where errors matter most.
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.uniform(-1, 1, (400_000, 2)), rng.uniform(-1, 1, (400_000, 2))], 1)
    band = s[_ground_truth().in_band(s, eps=0.1)][:20_000]
    sets["band"] = band

    records = []
    for name, states in sets.items():
        m = feasibility_metrics(critic, states)
        if hasattr(critic, "p_feasible"):
            p = _batched(critic.p_feasible, states, critic.device)
            m.update(calibration(p, check_feasible(states)))
        for k, v in m.items():
            records.append({"eval_set": name, "metric": k, "value": float(v)})
    return records


def empirical_infeasible_labels(dataset: OfflineDataset, h_fn, horizon: int = 20) -> np.ndarray:
    """True where the state's trajectory reaches h > 0 within `horizon` steps."""
    traj = split_trajectories(dataset)
    nxt = dataset.data["next_observations"]
    unsafe = _batched(h_fn, nxt.cpu().numpy(), nxt.device) > 0
    label = unsafe.copy()
    for j in range(1, horizon):
        ahead = np.zeros_like(unsafe)
        ahead[:-j] = unsafe[j:] & (traj[j:] == traj[:-j])
        label |= ahead
    return label


def evaluate_dsrl_proxy(critic, full_real: OfflineDataset, h_fn, horizon: int = 20, max_states: int = 200_000) -> List[dict]:
    from sklearn.metrics import roc_auc_score

    labels = empirical_infeasible_labels(full_real, h_fn, horizon)
    obs = full_real.data["observations"].cpu().numpy()
    if len(obs) > max_states:
        idx = np.random.default_rng(0).choice(len(obs), max_states, replace=False)
        obs, labels = obs[idx], labels[idx]
    v = _batched(critic.v_h, obs, critic.device)
    pred_infeasible = v > 0
    m = {
        "pred_feasible_frac": float((~pred_infeasible).mean()),
        "proxy_infeasible_frac": float(labels.mean()),
        "proxy_auroc": float(roc_auc_score(labels, v)) if 0 < labels.mean() < 1 else float("nan"),
        # predicted feasible although the trajectory violated soon after (unsafe mistake)
        "proxy_false_feasible_rate": float((~pred_infeasible & labels).sum() / max(labels.sum(), 1)),
        "proxy_false_infeasible_rate": float((pred_infeasible & ~labels).sum() / max((~labels).sum(), 1)),
    }
    if hasattr(critic, "p_feasible"):
        m.update({"proxy_" + k: v for k, v in calibration(_batched(critic.p_feasible, obs, critic.device), ~labels).items()})
    return [{"eval_set": "real_proxy", "metric": k, "value": float(x)} for k, x in m.items()]
