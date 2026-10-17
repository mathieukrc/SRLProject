import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from data.offline_dataset import OfflineDataset
from data.variant import split_trajectories

from .ensemble import EnsembleDynamics
from .normalizer import Normalizer
from .utils import make_dynamics_xy


@dataclass
class DynamicsTrainConfig:
    num_members: int = 5
    hidden_dims: Sequence[int] = (200, 200, 200)
    probabilistic: bool = False  # True for the DSRL tasks
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 20  # epochs without improvement, per member
    min_improvement: float = 1e-4  # relative improvement that counts
    bound_margin: float = 0.05  # clamp predictions to data range +- margin * range
    logvar_bound_coef: float = 0.01  # keeps the learned log-variance bounds tight
    eval_batch_size: int = 8192
    seed: int = 0


def split_by_trajectory(
    dataset: OfflineDataset, val_fraction: float = 0.15, seed: int = 0
) -> Tuple[OfflineDataset, OfflineDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    traj = split_trajectories(dataset)
    n_traj = int(traj[-1]) + 1
    if n_traj < 2:
        raise ValueError("need at least two trajectories to split")
    n_val = max(1, int(round(n_traj * val_fraction)))
    val_ids = np.random.default_rng(seed).permutation(n_traj)[:n_val]
    is_val = np.isin(traj, val_ids)
    return dataset.subset(np.flatnonzero(~is_val)), dataset.subset(
        np.flatnonzero(is_val)
    )


def data_bounds(
    dataset: OfflineDataset, margin: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = torch.cat(
        [dataset.data["observations"], dataset.data["next_observations"]], dim=0
    )
    lo, hi = states.min(dim=0).values, states.max(dim=0).values
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def _member_losses(
    model: EnsembleDynamics,
    xb: torch.Tensor,
    yb: torch.Tensor,
    cfg: DynamicsTrainConfig,
) -> torch.Tensor:
    """Training loss per member, shape (K,). xb: (K, B, in), yb: (K, B, d)."""
    mean, logvar = model.forward_normalized(xb)
    if logvar is None:
        return ((mean - yb) ** 2).mean(dim=(1, 2))
    nll = ((mean - yb) ** 2) * torch.exp(-logvar) + logvar
    return nll.mean(dim=(1, 2))


@torch.no_grad()
def _val_mse(
    model: EnsembleDynamics, x: torch.Tensor, y: torch.Tensor, chunk: int
) -> torch.Tensor:
    total = torch.zeros(model.num_members, device=x.device)
    for i in range(0, len(x), chunk):
        mean, _ = model.forward_normalized(x[i : i + chunk])
        total += ((mean - y[i : i + chunk]) ** 2).sum(dim=(1, 2))
    return total / (len(x) * y.shape[-1])


def train_dynamics_ensemble(
    train_ds: OfflineDataset,
    val_ds: OfflineDataset,
    cfg: DynamicsTrainConfig = DynamicsTrainConfig(),
    obs_low: Optional[torch.Tensor] = None,
    obs_high: Optional[torch.Tensor] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> EnsembleDynamics:
    device = train_ds.device
    torch.manual_seed(cfg.seed)  # member initialisations
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    x_tr, y_tr = make_dynamics_xy(train_ds)
    x_va, y_va = make_dynamics_xy(val_ds)
    obs_dim = train_ds.data["observations"].shape[-1]
    act_dim = train_ds.data["actions"].shape[-1]

    in_norm, out_norm = Normalizer(x_tr).to(device), Normalizer(y_tr).to(device)

    lo, hi = data_bounds(train_ds, cfg.bound_margin)
    lo = lo if obs_low is None else torch.as_tensor(obs_low, dtype=torch.float32)
    hi = hi if obs_high is None else torch.as_tensor(obs_high, dtype=torch.float32)

    model = EnsembleDynamics(
        obs_dim,
        act_dim,
        num_members=cfg.num_members,
        hidden_dims=cfg.hidden_dims,
        probabilistic=cfg.probabilistic,
        in_norm=in_norm,
        out_norm=out_norm,
        obs_low=lo,
        obs_high=hi,
    ).to(device)

    xn, yn = in_norm.normalize(x_tr), out_norm.normalize(y_tr)
    xv, yv = in_norm.normalize(x_va), out_norm.normalize(y_va)

    K, N = cfg.num_members, len(xn)
    boot = torch.randint(N, (K, N), device=device, generator=gen)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best = torch.full((K,), float("inf"), device=device)
    stale = torch.zeros(K, dtype=torch.long, device=device)
    best_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    steps_per_epoch = math.ceil(N / cfg.batch_size)

    for epoch in range(cfg.max_epochs):
        model.train()
        running = torch.zeros(K, device=device)
        for _ in range(steps_per_epoch):
            pos = torch.randint(N, (K, cfg.batch_size), device=device, generator=gen)
            idx = boot.gather(1, pos)  # (K, B): each member's own minibatch
            losses = _member_losses(model, xn[idx], yn[idx], cfg)
            loss = losses.sum()
            if model.probabilistic:
                loss = loss + cfg.logvar_bound_coef * (
                    model.max_logvar.sum() - model.min_logvar.sum()
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += losses.detach()

        model.eval()
        val = _val_mse(model, xv, yv, cfg.eval_batch_size)

        improved = val < best * (1.0 - cfg.min_improvement)
        if improved.any():
            for n, p in model.named_parameters():  # every parameter has leading dim K
                best_params[n][improved] = p.detach()[improved]
            best = torch.where(improved, val, best)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)

        if log_fn is not None and (
            epoch % 10 == 0 or bool((stale >= cfg.patience).all())
        ):
            log_fn(
                f"epoch {epoch:4d}  train {(running / steps_per_epoch).mean():.5f}  "
                f"val {val.mean():.5f}  best {best.mean():.5f}  "
                f"stopped {int((stale >= cfg.patience).sum())}/{K}"
            )
        if bool((stale >= cfg.patience).all()):
            break

    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(best_params[n])

    model.metadata.update(
        train_config=asdict(cfg),
        val_mse_per_member=best.cpu().tolist(),
        n_train=int(N),
        n_val=int(len(xv)),
    )
    return model.eval()


@torch.no_grad()
def disagreement_on(
    model: EnsembleDynamics, obs: torch.Tensor, act: torch.Tensor, chunk: int = 65536
) -> np.ndarray:
    return (
        torch.cat(
            [
                model.disagreement(obs[i : i + chunk], act[i : i + chunk])
                for i in range(0, len(obs), chunk)
            ]
        )
        .cpu()
        .numpy()
    )


def calibrate_threshold(
    model: EnsembleDynamics, train_ds: OfflineDataset, percentile: float
) -> float:
    """Sets the threshold at a percentile of disagreement on the TRAINING transitions.
    Store the percentile: that is what carries over between environments."""
    d = disagreement_on(model, train_ds.data["observations"], train_ds.data["actions"])
    threshold = float(np.percentile(d, percentile))
    model.disagreement_threshold = threshold
    model.metadata.update(threshold_percentile=float(percentile), threshold=threshold)
    return threshold


def train_or_load(
    path: str,
    dataset: OfflineDataset,
    metadata: Dict[str, Any],
    cfg: DynamicsTrainConfig = DynamicsTrainConfig(),
    val_fraction: float = 0.15,
    percentile: float = 95.0,
    obs_low: Optional[torch.Tensor] = None,
    obs_high: Optional[torch.Tensor] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> EnsembleDynamics:
    """
    metadata must identify the training data, e.g. {"env": "controlled",
    "variant": "cov25", "seed": 0}. Loading checks it, so an ensemble trained on one
    coverage variant can never be used for another.
    """
    if os.path.exists(path):
        return EnsembleDynamics.load(path, device=str(dataset.device), expect=metadata)

    train_ds, val_ds = split_by_trajectory(dataset, val_fraction, seed=cfg.seed)
    model = train_dynamics_ensemble(train_ds, val_ds, cfg, obs_low, obs_high, log_fn)
    calibrate_threshold(model, train_ds, percentile)
    model.metadata.update(metadata)
    model.save(path)
    return model
