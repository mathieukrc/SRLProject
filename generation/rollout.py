"""
Synthetic chains: roll every (start, first action) branch forward for up to k steps.

Filter F, applied at every step:
    - non-finite predictions are dropped;
    - with a learned model, a step whose ensemble disagreement exceeds the calibrated
      threshold is dropped and the chain stops there;
    - a next state that is not valid (valid_fn) is dropped and the chain stops, unless it
      is terminal (e.g. leaving the arena), which is kept because it carries cost.

Masks: chain ends are truncations, not terminations, so timeouts are always 0 and the
critic bootstraps from V_h at the last state. Only terminal_fn produces terminals.
"""

from typing import Dict

import numpy as np
import torch

from .actions import h_greedy_actions
from .context import GenContext


@torch.no_grad()
def rollout_chains(
    starts: torch.Tensor,
    first_actions: torch.Tensor,
    ctx: GenContext,
    chain_len: int = 5,
    policy_noise: float = 0.1,
    n_candidates: int = 32,
    horizon: int = 5,
) -> Dict[str, np.ndarray]:
    S, B, A = first_actions.shape
    obs = starts.repeat_interleave(B, dim=0)
    act = first_actions.reshape(S * B, A)
    alive = torch.ones(len(obs), dtype=torch.bool, device=obs.device)
    chain_id = torch.arange(len(obs), device=obs.device)

    rows = {k: [] for k in ("observations", "actions", "next_observations", "terminals", "chain")}
    for t in range(chain_len):
        if t > 0:
            greedy = h_greedy_actions(obs, ctx, n_candidates, horizon)
            eps = torch.randn(greedy.shape, device=obs.device, generator=ctx.generator)
            act = ctx.clip_actions(greedy + policy_noise * eps * (ctx.act_high - ctx.act_low) / 2)

        nxt, dis = ctx.step_model.step(obs, act)
        finite = torch.isfinite(nxt).all(dim=-1)
        nxt = torch.where(finite.unsqueeze(-1), nxt, obs)  # keep shapes; rows are masked out

        terminal = ctx.terminal_fn(nxt).reshape(-1).bool()
        ok = alive & finite & (terminal | ctx.valid_fn(nxt).reshape(-1).bool())
        if dis is not None:
            ok &= dis <= ctx.step_model.threshold

        if ok.any():
            rows["observations"].append(obs[ok])
            rows["actions"].append(act[ok])
            rows["next_observations"].append(nxt[ok])
            rows["terminals"].append(terminal[ok].float())
            rows["chain"].append(chain_id[ok])

        alive = ok & ~terminal
        obs = nxt
        if not alive.any():
            break

    if not rows["observations"]:
        return empty_raw(starts.shape[1], A)

    out = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
    n = len(out["observations"])
    out["actions"] = out["actions"].astype(np.float32)
    out["rewards"] = np.zeros(n, dtype=np.float32)  # critic-only augmentation: unused
    out["costs"] = np.zeros(n, dtype=np.float32)  # recomputed from h_fn by the dataset
    out["timeouts"] = np.zeros(n, dtype=np.float32)
    return out


def empty_raw(obs_dim: int, act_dim: int) -> Dict[str, np.ndarray]:
    return {
        "observations": np.zeros((0, obs_dim), np.float32),
        "actions": np.zeros((0, act_dim), np.float32),
        "next_observations": np.zeros((0, obs_dim), np.float32),
        "rewards": np.zeros(0, np.float32),
        "costs": np.zeros(0, np.float32),
        "terminals": np.zeros(0, np.float32),
        "timeouts": np.zeros(0, np.float32),
        "chain": np.zeros(0, np.int64),
    }


def concat_raw(parts) -> Dict[str, np.ndarray]:
    parts = [p for p in parts if len(p["observations"])]
    if not parts:
        raise ValueError("concat_raw needs at least one non-empty part")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def take_raw(raw: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    """First n rows. Cutting a chain early is just an earlier truncation (mask stays 1)."""
    return {k: v[:n] for k, v in raw.items()}
