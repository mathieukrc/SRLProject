"""
Policy-level evaluation: roll out the trained FISOR agent in the real environment.

Works with any env exposing the gymnasium API (reset() -> (obs, info),
step(a) -> (obs, reward, terminated, truncated, info) with info["cost"]).
For DSRL envs the normalised reward/cost of the benchmark are reported as well.
"""

from typing import Callable, List, Optional

import numpy as np


def _find_offline_env(env):
    """DSRL keeps get_normalized_score on an inner wrapper; walk the wrapper chain."""
    e = env
    for _ in range(20):
        if hasattr(e, "get_normalized_score") and hasattr(e, "set_target_cost"):
            return e
        if not hasattr(e, "env"):
            break
        e = e.env
    return None


def evaluate_policy(
    agent,
    make_env: Callable[[int], object],
    n_episodes: int = 10,
    target_cost: Optional[float] = None,
    seed: int = 0,
) -> List[dict]:
    rewards, costs, lengths = [], [], []
    env = make_env(seed)
    offline = _find_offline_env(env)
    if offline is not None and target_cost is not None:
        offline.set_target_cost(target_cost)

    for ep in range(n_episodes):
        try:
            obs, _ = env.reset(seed=seed + ep)
        except TypeError:  # envs whose reset() takes no seed
            obs, _ = env.reset()
        ep_r, ep_c, t, done = 0.0, 0.0, 0, False
        while not done:
            act = agent.act(np.asarray(obs, dtype=np.float32))
            obs, r, terminated, truncated, info = env.step(act)
            ep_r += float(r)
            ep_c += float(info.get("cost", 0.0))
            t += 1
            done = bool(terminated or truncated)
        rewards.append(ep_r)
        costs.append(ep_c)
        lengths.append(t)

    rec = {
        "return": float(np.mean(rewards)),
        "return_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "episode_len": float(np.mean(lengths)),
        "frac_episodes_with_cost": float(np.mean(np.asarray(costs) > 0)),
    }
    if offline is not None and target_cost is not None:
        nr, nc = zip(*(offline.get_normalized_score(r, c) for r, c in zip(rewards, costs)))
        rec["normalized_return"] = float(np.mean(nr))
        rec["normalized_cost"] = float(np.mean(nc))
    return [{"eval_set": "policy", "metric": k, "value": v} for k, v in rec.items()]
