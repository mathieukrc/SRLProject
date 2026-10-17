"""
Bootstrap ensemble of dynamics models (predicts normalised state deltas).
Member disagreement is what the synthetic-data filter thresholds on.
Train it ONLY on the dataset variant of the run.
"""

import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from architectures.vectorized_mlp import VectorizedMLP
from data.offline_dataset import OfflineDataset
from dynamics.normalizer import Normalizer
from dynamics.utils import make_dynamics_xy


class DynamicsEnsemble(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_members: int,
        in_norm: Normalizer,
        out_norm: Normalizer,
        hidden_dims: Tuple[int, ...] = (200, 200, 200),
    ) -> None:
        super().__init__()
        self.obs_dim, self.act_dim, self.n_members = obs_dim, act_dim, n_members
        self.hidden_dims = tuple(hidden_dims)
        self.net = VectorizedMLP(
            obs_dim + act_dim, n_members, out_dim=obs_dim, hidden_dims=hidden_dims
        )
        self.in_norm = in_norm
        self.out_norm = out_norm
        # 95th percentile of disagreement on real validation data; set after training.
        self.register_buffer("disagreement_scale", torch.tensor(1.0))

    def forward_normalised(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Normalised delta predictions, (K, B, obs_dim)."""
        x = self.in_norm.normalize(torch.cat((obs, act), dim=-1))
        return self.net(x)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Next-state predictions of every member, (K, B, obs_dim)."""
        return obs.unsqueeze(0) + self.out_norm.denormalize(self.forward_normalised(obs, act))

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        member: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (next_obs (B, d), disagreement (B,)) in normalised delta units."""
        z = self.forward_normalised(obs, act)  # (K, B, d)
        mean = z.mean(dim=0)
        disagreement = (z - mean).norm(dim=-1).max(dim=0).values
        if member is None:
            z_sel = mean
        else:
            z_sel = z[member, torch.arange(z.shape[1], device=z.device)]
        return obs + self.out_norm.denormalize(z_sel), disagreement


def _batched_disagreement(model: DynamicsEnsemble, x_obs, x_act, chunk=65536) -> torch.Tensor:
    out = []
    for i in range(0, len(x_obs), chunk):
        out.append(model.predict(x_obs[i : i + chunk], x_act[i : i + chunk])[1])
    return torch.cat(out)


def train_dynamics_ensemble(
    train: OfflineDataset,
    val: OfflineDataset,
    n_members: int = 5,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 1024,
    patience: Optional[int] = 5,
    seed: int = 0,
    max_steps_per_epoch: int = 500,
    verbose: bool = True,
) -> DynamicsEnsemble:
    device = train.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    torch.manual_seed(seed)

    x_tr, y_tr = make_dynamics_xy(train)
    x_va, y_va = make_dynamics_xy(val)
    obs_dim = train.data["observations"].shape[-1]
    act_dim = train.data["actions"].shape[-1]

    in_norm = Normalizer(x_tr).to(device)
    out_norm = Normalizer(y_tr).to(device)
    model = DynamicsEnsemble(obs_dim, act_dim, n_members, in_norm, out_norm).to(device)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=1e-5)

    xn_tr, yn_tr = in_norm.normalize(x_tr), out_norm.normalize(y_tr)
    xn_va, yn_va = in_norm.normalize(x_va), out_norm.normalize(y_va)
    n = len(xn_tr)
    boot = torch.poisson(torch.ones((n_members, n), device=device), generator=g)

    def val_loss() -> float:
        with torch.no_grad():
            losses = []
            for i in range(0, len(xn_va), 65536):
                pred = model.net(xn_va[i : i + 65536])
                losses.append(((pred - yn_va[i : i + 65536]) ** 2).mean(dim=(0, 2)).sum())
            return float(torch.stack(losses).sum() / max(1, len(losses)))

    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    steps = min(max_steps_per_epoch, max(1, n // batch_size))
    for epoch in range(num_epochs):
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=device, generator=g)
            pred = model.net(xn_tr[idx])  # (K, B, d)
            per = ((pred - yn_tr[idx]) ** 2).mean(dim=-1)  # (K, B)
            w = boot[:, idx]
            loss = ((per * w).sum(-1) / w.sum(-1).clamp_min(1e-8)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = val_loss()
        if verbose:
            print(f"  dynamics epoch {epoch + 1}/{num_epochs}: val {vl:.5f} (best {best:.5f})")
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    d = _batched_disagreement(model, val.data["observations"], val.data["actions"])
    model.disagreement_scale.fill_(float(torch.quantile(d.float().cpu(), 0.95).clamp_min(1e-8)))
    return model


@torch.no_grad()
def dynamics_validation_report(model: DynamicsEnsemble, val: OfflineDataset) -> dict:
    obs, act = val.data["observations"], val.data["actions"]
    nxt = val.data["next_observations"]
    preds, dis = [], []
    for i in range(0, len(obs), 65536):
        p, d = model.predict(obs[i : i + 65536], act[i : i + 65536])
        preds.append(p)
        dis.append(d)
    pred, dis = torch.cat(preds), torch.cat(dis)
    err = (model.out_norm.normalize(pred - obs) - model.out_norm.normalize(nxt - obs)).norm(dim=-1)
    e, d = err.cpu().numpy(), dis.cpu().numpy()
    corr = float(np.corrcoef(e, d)[0, 1]) if e.std() > 0 and d.std() > 0 else float("nan")
    return {
        "val_mse_normalised": float((err ** 2).mean()),
        "val_err_p95": float(np.quantile(e, 0.95)),
        "disagreement_p95": float(model.disagreement_scale),
        "corr_disagreement_error": corr,
    }


def save_dynamics(model: DynamicsEnsemble, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "dims": (model.obs_dim, model.act_dim, model.n_members, model.hidden_dims),
        "norm_shapes": (model.in_norm.mean.shape[0], model.out_norm.mean.shape[0]),
        "state": model.state_dict(),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_dynamics(path: str, device: str = "cpu") -> DynamicsEnsemble:
    payload = torch.load(path, map_location=device)
    obs_dim, act_dim, k, hidden = payload["dims"]
    n_in, n_out = payload["norm_shapes"]
    in_norm = Normalizer(torch.zeros(2, n_in))
    out_norm = Normalizer(torch.zeros(2, n_out))
    model = DynamicsEnsemble(obs_dim, act_dim, k, in_norm, out_norm, tuple(hidden))
    model.load_state_dict(payload["state"])
    return model.to(device).eval()
