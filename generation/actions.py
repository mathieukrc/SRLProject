"""
Action proposal for synthetic chains.

FISOR's V_h is a low expectile over the actions that appear near a state. A start state
that only gets one (random) action is therefore labelled by that single action: if it
happens to run into an obstacle, the state looks infeasible even when it is not. So every
start state gets B actions ("branches"):

    branch 0       h-greedy: the candidate action whose short constant-action lookahead
                   reaches the lowest max h. This is where the known cost structure is used
                   to produce recovery behaviour the dataset may lack.
    branch 1       action of the nearest real state, plus noise (stays near data support).
    branches 2..   uniform random actions.
"""

import torch

from .context import GenContext


@torch.no_grad()
def h_greedy_actions(
    obs: torch.Tensor,
    ctx: GenContext,
    n_candidates: int = 32,
    horizon: int = 5,
    chunk: int = 4096,
) -> torch.Tensor:
    """For each state, the candidate action with the lowest max-h over a constant-action
    lookahead of `horizon` steps through ctx.step_model. Shape (N, act_dim)."""
    out = []
    for i in range(0, len(obs), chunk):
        o = obs[i : i + chunk]
        n = len(o)
        cand = ctx.uniform_actions(n * n_candidates)  # (n*M, A)
        s = o.repeat_interleave(n_candidates, dim=0)
        score = ctx.h_fn(s)
        for _ in range(horizon):
            s, dis = ctx.step_model.step(s, cand)
            score = torch.maximum(score, ctx.h_fn(s))
            if dis is not None:  # do not trust lookaheads the model is unsure about
                score = score + (dis > ctx.step_model.threshold).float() * 1e3
        best = score.view(n, n_candidates).argmin(dim=1)
        out.append(cand.view(n, n_candidates, -1)[torch.arange(n, device=o.device), best])
    return torch.cat(out)


@torch.no_grad()
def nearest_neighbour_actions(
    obs: torch.Tensor, ctx: GenContext, noise: float = 0.1
) -> torch.Tensor:
    tree, scale, rows = ctx.nn_index()
    _, idx = tree.query(obs.detach().cpu().numpy() / scale, k=1)
    act = ctx.data_act[torch.as_tensor(rows[idx], device=ctx.data_act.device)]
    act = act.to(obs.device)
    eps = torch.randn(act.shape, device=obs.device, generator=ctx.generator)
    return ctx.clip_actions(act + noise * eps * (ctx.act_high - ctx.act_low) / 2)


@torch.no_grad()
def propose_first_actions(
    starts: torch.Tensor,
    ctx: GenContext,
    branching: int = 4,
    n_candidates: int = 32,
    horizon: int = 5,
    nn_noise: float = 0.1,
) -> torch.Tensor:
    """Shape (S, branching, act_dim)."""
    if branching < 1:
        raise ValueError("branching must be >= 1")
    S = len(starts)
    branches = [h_greedy_actions(starts, ctx, n_candidates, horizon)]
    if branching >= 2:
        branches.append(nearest_neighbour_actions(starts, ctx, nn_noise))
    for _ in range(branching - len(branches)):
        branches.append(ctx.uniform_actions(S))
    return torch.stack(branches[:branching], dim=1)
