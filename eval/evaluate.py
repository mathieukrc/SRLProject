"""
Evaluation:
  1. feasibility estimate vs ground truth   (controlled env)
  2. feasibility estimate vs proxies        (all envs)
  3. deployed policy return / cost
plus calibration of the uncertainty signal.

Evaluation sets are built ONCE per env from the FULL dataset (stage A) and cached,
so every run is scored on identical states.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from data.offline_dataset import OfflineDataset
from data.utils import HFn
from envs.registry import EnvSpec


def _eval_path(env: str) -> str:
    from experiments.paths import cache_dir

    return str(cache_dir() / "eval" / f"{env}.npz")


def _precursor_mask(dataset: OfflineDataset, h_fn, k: int) -> np.ndarray:
    """Transitions within k steps before a violation in the same trajectory."""
    from data.variant import split_trajectories

    obs = dataset.data["observations"]
    with torch.no_grad():
        unsafe = (h_fn(dataset.data["next_observations"]).reshape(-1) > 0).cpu().numpy()
    traj = split_trajectories(dataset)
    pre = unsafe.copy()  # j = 0: the next state itself is unsafe
    for j in range(1, k + 1):
        ahead = np.zeros_like(unsafe)
        ahead[:-j] = unsafe[j:] & (traj[j:] == traj[:-j])
        pre |= ahead
    with torch.no_grad():
        h_now = h_fn(obs).reshape(-1).cpu().numpy()
    return pre & (h_now <= 0)


def build_evaluation_states(spec: EnvSpec, full: OfflineDataset, seed: int = 0,
                            n_max: int = 20000, precursor_steps: int = 5) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = full.data["observations"].cpu().numpy()
    with torch.no_grad():
        h_obs = spec.h_fn(full.data["observations"]).reshape(-1).cpu().numpy()
    pre = _precursor_mask(full, spec.h_fn, precursor_steps)

    def sample(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        return obs[rng.choice(idx, size=min(n_max, len(idx)), replace=False)] if len(idx) else obs[:0]

    sets = {
        "proxy_dataset": sample(np.ones(len(obs), dtype=bool)),
        "proxy_unsafe": sample(h_obs > 0),
        "proxy_precursor": sample(pre),
    }
    if spec.has_ground_truth:
        import controlled.controlled_env as ce
        from controlled.env_api import _ground_truth
        from test.metrics import evaluation_sets_controlled_env

        for k, v in evaluation_sets_controlled_env(full, n_random=n_max, seed=seed).items():
            sets[k] = v[rng.choice(len(v), size=min(len(v), n_max), replace=False)] if len(v) > n_max else v
        gt = _ground_truth()
        cand = np.concatenate([rng.uniform(-1, 1, (40 * n_max, 2)), rng.uniform(-1, 1, (40 * n_max, 2))], 1)
        cand = cand[(ce.h_fn(cand) <= 0) & gt.in_band(cand)]
        sets["band"] = cand[:n_max]
    return {k: np.asarray(v, dtype=np.float32) for k, v in sets.items()}


def evaluation_states(spec: EnvSpec, full: Optional[OfflineDataset] = None) -> Dict[str, np.ndarray]:
    """Load the cached sets; build them from `full` if missing (stage A does this)."""
    path = _eval_path(spec.name)
    if not os.path.exists(path):
        if full is None:
            raise FileNotFoundError(f"{path} missing: run `experiments.prepare --stage data` first")
        sets = build_evaluation_states(spec, full)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **sets)
        os.replace(tmp, path)
    d = np.load(path)
    return {k: d[k] for k in d.files}


@torch.no_grad()
def _v(critic, states: np.ndarray, chunk: int = 65536) -> torch.Tensor:
    out = []
    for i in range(0, len(states), chunk):
        out.append(critic.v_h(torch.as_tensor(states[i : i + chunk], device=critic.device)))
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def evaluate_feasibility(critic, spec: EnvSpec, eval_states: Dict[str, np.ndarray],
                         training_h: HFn) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if spec.has_ground_truth:
        from test.metrics import feasibility_metrics

        for name in ("dataset", "random", "slice_v=(0.6,0)", "band"):
            if name in eval_states and len(eval_states[name]):
                for k, v in feasibility_metrics(critic, eval_states[name]).items():
                    out[f"{name}/{k}"] = v

    def frac_feasible(name: str) -> float:
        s = eval_states.get(name)
        if s is None or len(s) == 0:
            return float("nan")
        return float((_v(critic, s) <= 0).float().mean())

    out["proxy/unsafe_called_feasible"] = frac_feasible("proxy_unsafe")
    out["proxy/precursor_called_feasible"] = frac_feasible("proxy_precursor")
    s = eval_states["proxy_dataset"]
    v = _v(critic, s)
    h = training_h(torch.as_tensor(s, device=critic.device)).reshape(-1)
    out["proxy/lower_bound_violation"] = float((v < h - 1e-6).float().mean())
    out["proxy/pred_feasible_frac"] = float((v <= 0).float().mean())
    return out


@torch.no_grad()
def uncertainty_calibration(critic, states: np.ndarray, true_feasible: np.ndarray,
                            n_bins: int = 10) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=critic.device)
    unc = critic.feasibility_entropy(s).cpu().numpy()
    std = critic.v_h_members(s).std(dim=0).cpu().numpy()
    pred = critic.is_feasible(s).cpu().numpy()
    err = (pred != np.asarray(true_feasible)).astype(float)

    def auroc(score):
        return float(roc_auc_score(err, score)) if 0 < err.mean() < 1 else float("nan")

    def rho(score):
        r = spearmanr(score, err).correlation
        return float(r) if np.isfinite(r) else float("nan")

    out: Dict[str, Any] = {
        "error_rate": float(err.mean()),
        "auroc_entropy": auroc(unc), "spearman_entropy": rho(unc),
        "auroc_std": auroc(std), "spearman_std": rho(std),
    }
    edges = np.unique(np.quantile(std, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, std, side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = which == b
        out[f"bin{b}_std"] = float(std[m].mean()) if m.any() else float("nan")
        out[f"bin{b}_error"] = float(err[m].mean()) if m.any() else float("nan")
    return out


@torch.no_grad()
def feasibility_map(critic, velocity=(0.6, 0.0), resolution: float = 0.01,
                    quantity: str = "v_h") -> np.ndarray:
    """(n, n) array indexed [y, x] over [-1, 1)^2 (same layout as main.py's plot)."""
    g = np.arange(-1.0, 1.0, resolution, dtype=np.float32)
    X, Y = np.meshgrid(g, g, indexing="xy")
    S = np.stack([X.ravel(), Y.ravel(), np.full(X.size, velocity[0]), np.full(X.size, velocity[1])], 1)
    s = torch.as_tensor(S.astype(np.float32), device=critic.device)
    if quantity == "v_h":
        v = critic.v_h(s)
    elif quantity == "p_feasible":
        v = critic.p_feasible(s)
    elif quantity == "entropy":
        v = critic.feasibility_entropy(s)
    else:
        raise ValueError(quantity)
    return v.cpu().numpy().reshape(X.shape)


@torch.no_grad()
def evaluate_policy(agent, spec: EnvSpec, n_episodes: int, seed: int) -> Dict[str, float]:
    rets, costs, viol = [], [], []
    if spec.name == "controlled":
        from controlled.controlled_env import ControlledEnv

        env = ControlledEnv(seed=seed)
        for _ in range(n_episodes):
            s, done, R, C = env.reset(), False, 0.0, 0.0
            while not done:
                s, r, c, done, _ = env.step(agent.act(s.astype(np.float32)))
                R, C = R + r, C + c
            rets.append(R), costs.append(C), viol.append(C > 0)
        return {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
                "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                "violation_episode_rate": float(np.mean(viol))}

    import dsrl  # noqa: F401  (registers the Offline* envs)
    import gymnasium as gym

    env = gym.make(spec.dsrl_id)
    norm_r, norm_c = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, R, C = False, 0.0, 0.0
        while not done:
            obs, r, terminated, truncated, info = env.step(agent.act(np.asarray(obs, dtype=np.float32)))
            R, C = R + float(r), C + float(info.get("cost", 0.0))
            done = terminated or truncated
        rets.append(R), costs.append(C), viol.append(C > 0)
        try:
            nr, nc = env.get_normalized_score(R, C)
            norm_r.append(float(nr)), norm_c.append(float(nc))
        except Exception:
            pass
    out = {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
           "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
           "violation_episode_rate": float(np.mean(viol))}
    if norm_r:
        out["normalized_return"] = float(np.mean(norm_r))
        out["normalized_cost"] = float(np.mean(norm_c))
    return out
