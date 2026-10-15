from typing import Dict

import numpy as np
import torch

import controlled.controlled_env as ce
from controlled.env_api import _ground_truth, check_feasible
from data.offline_dataset import OfflineDataset


def evaluation_sets_controlled_env(
    dataset: OfflineDataset, n_random: int = 20000, seed: int = 0
) -> Dict[str, np.ndarray]:
    """Three groups of test states, each of shape (N, 4)."""
    rng = np.random.default_rng(seed)

    data_states = dataset.data["observations"].cpu().numpy()

    s = np.concatenate(
        [rng.uniform(-1, 1, (4 * n_random, 2)), rng.uniform(-1, 1, (4 * n_random, 2))],
        axis=1,
    )
    random_states = s[ce.h_fn(s) <= 0][:n_random]

    g = np.arange(-1.0, 1.0, 0.01)
    X, Y = np.meshgrid(g, g, indexing="ij")
    slice_states = np.stack(
        [X.ravel(), Y.ravel(), np.full(X.size, 0.6), np.zeros(X.size)], axis=1
    )

    return {
        "dataset": data_states,
        "random": random_states,
        "slice_v=(0.6,0)": slice_states,
    }


@torch.no_grad()
def feasibility_metrics(critic, states: np.ndarray) -> Dict[str, float]:
    states = np.asarray(states, dtype=np.float32)
    pred = (
        critic.is_feasible(torch.as_tensor(states, device=critic.device)).cpu().numpy()
    )
    true = check_feasible(states)
    band = _ground_truth().in_band(states)

    tp = np.sum(pred & true)  # feasible, predicted feasible
    tn = np.sum(~pred & ~true)  # infeasible, predicted infeasible
    fp = np.sum(pred & ~true)  # infeasible, predicted feasible  -> unsafe mistake
    fn = np.sum(~pred & true)  # feasible, predicted infeasible  -> overly conservative

    def rate(a, b):
        return float(a / b) if b > 0 else float("nan")

    return {
        "accuracy": rate(tp + tn, len(true)),
        "balanced_accuracy": 0.5 * (rate(tp, tp + fn) + rate(tn, tn + fp)),
        "false_feasible_rate": rate(
            fp, fp + tn
        ),  # share of infeasible states called feasible
        "false_infeasible_rate": rate(
            fn, fn + tp
        ),  # share of feasible states called infeasible
        "iou_feasible": rate(tp, tp + fp + fn),
        "band_accuracy": rate(np.sum((pred == true) & band), band.sum()),
        "pred_feasible_frac": float(pred.mean()),
        "true_feasible_frac": float(true.mean()),
    }
