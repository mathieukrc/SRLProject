"""
Policy-level evaluation: roll out the trained FISOR agent in the real environment.

The benchmark environments do not agree on the step() API:
    Bullet-Safety-Gym (CarCircle)    (obs, reward, terminated, truncated, info), cost in info
    safety-gymnasium  (PointGoal)    (obs, reward, cost, terminated, truncated, info)
gym.make() wraps both in gymnasium's TimeLimit / OrderEnforcing / PassiveEnvChecker,
which only accept the first form and crash on the second. So we step the innermost
environment directly, accept both forms, and enforce the time limit ourselves.

For DSRL envs the benchmark's normalised reward and cost are reported as well.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np


def _find_offline_env(env):
    """DSRL keeps get_normalized_score on one of the wrappers; walk the chain."""
    e = env
    for _ in range(20):
        if hasattr(e, "get_normalized_score") and hasattr(e, "set_target_cost"):
            return e
        if not hasattr(e, "env"):
            break
        e = e.env
    return None


def _max_steps(env) -> Optional[int]:
    spec = getattr(env, "spec", None)
    return getattr(spec, "max_episode_steps", None)


def _step(env, action) -> Tuple[np.ndarray, float, float, bool, bool, dict]:
    out = env.step(action)
    if len(out) == 6:  # safety-gymnasium
        obs, r, c, terminated, truncated, info = out
    elif len(out) == 5:  # gymnasium
        obs, r, terminated, truncated, info = out
        c = info.get("cost", 0.0)
    elif len(out) == 4:  # old gym
        obs, r, done, info = out
        c, terminated, truncated = info.get("cost", 0.0), done, False
    else:
        raise ValueError(f"step() returned {len(out)} values")
    return obs, float(r), float(c), bool(terminated), bool(truncated), info


def evaluate_policy(
    agent,
    make_env: Callable[[int], object],
    n_episodes: int = 10,
    target_cost: Optional[float] = None,
    seed: int = 0,
    default_max_steps: int = 1000,
) -> List[dict]:
    outer = make_env(seed)
    offline = _find_offline_env(outer)
    if offline is not None and target_cost is not None:
        offline.set_target_cost(target_cost)

    # gymnasium envs: bypass the wrappers. Our own adapters (controlled env) have no
    # `unwrapped` and are used as they are.
    max_steps = _max_steps(outer)
    env = getattr(outer, "unwrapped", outer)
    if max_steps is None:
        max_steps = _max_steps(env) or default_max_steps

    rewards, costs, lengths = [], [], []
    for ep in range(n_episodes):
        try:
            obs, _ = env.reset(seed=seed + ep)
        except TypeError:  # envs whose reset() takes no seed
            obs, _ = env.reset()
        ep_r, ep_c, t, done = 0.0, 0.0, 0, False
        while not done:
            act = agent.act(np.asarray(obs, dtype=np.float32))
            obs, r, c, terminated, truncated, _ = _step(env, act)
            ep_r += r
            ep_c += c
            t += 1
            done = terminated or truncated or t >= max_steps
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
