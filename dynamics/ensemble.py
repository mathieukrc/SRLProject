"""
Ensemble of dynamics models, used for the DSRL tasks (where the true dynamics are not
available offline) and for the learned-dynamics ablation of the controlled environment.

Each member predicts the state delta from (s, a). Members differ by initialisation and
by Poisson bootstrap weights on the training data. The spread between members is the
disagreement signal used by the filter F: a synthetic rollout is truncated as soon as the
members disagree more than they do on real held-out transitions.
"""

import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from architectures.vectorized_mlp import VectorizedMLP
from data.offline_dataset import OfflineDataset

from .normalizer import Normalizer
from .utils import make_dynamics_xy


class DynamicsEnsemble(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        K: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.obs_dim, self.act_dim, self.K = obs_dim, act_dim, K
        self.hidden_dims = tuple(hidden_dims)
        self.net = VectorizedMLP(
            obs_dim + act_dim, ensemble_size=K, out_dim=obs_dim, hidden_dims=hidden_dims
        )
        # Filled in by fit(); registered as buffers so they move with .to() and are saved.
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))
        self.register_buffer("out_mean", torch.zeros(obs_dim))
        self.register_buffer("out_std", torch.ones(obs_dim))
        # Disagreement on held-out real transitions; rollouts are cut above this.
        self.register_buffer("threshold", torch.tensor(float("inf")))

    def _set_norm(self, x: torch.Tensor, y: torch.Tensor) -> None:
        nx, ny = Normalizer(x), Normalizer(y)
        self.in_mean.copy_(nx.mean)
        self.in_std.copy_(nx.std)
        self.out_mean.copy_(ny.mean)
        self.out_std.copy_(ny.std)

    def _raw(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Normalised (obs, act) -> normalised delta, shape (K, B, obs_dim)."""
        out = self.net(x_norm)
        if out.dim() == 2:  # obs_dim == 1 is squeezed by VectorizedMLP
            out = out.unsqueeze(-1)
        return out

    @torch.no_grad()
    def predict_all(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Next-state prediction of every member, shape (K, B, obs_dim)."""
        x = (torch.cat((obs, act), dim=-1) - self.in_mean) / self.in_std
        delta = self._raw(x) * self.out_std + self.out_mean
        return obs.unsqueeze(0) + delta

    @torch.no_grad()
    def disagreement_of(self, preds: torch.Tensor) -> torch.Tensor:
        """RMS (over dims) of the member std, in units of the delta's std. Shape (B,)."""
        z = preds / self.out_std
        return z.std(dim=0).pow(2).mean(dim=-1).sqrt()

    @torch.no_grad()
    def step(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean next state (B, obs_dim) and disagreement (B,)."""
        preds = self.predict_all(obs, act)
        return preds.mean(dim=0), self.disagreement_of(preds)

    # ------------------------------------------------------------------ training
    def fit(
        self,
        train: OfflineDataset,
        val: OfflineDataset,
        max_epochs: int = 200,
        steps_per_epoch: int = 500,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
        threshold_quantile: float = 0.99,
        seed: int = 0,
        log_fn=print,
    ) -> Dict[str, float]:
        device = self.in_mean.device
        x_tr, y_tr = (t.to(device) for t in make_dynamics_xy(train))
        x_va, y_va = (t.to(device) for t in make_dynamics_xy(val))
        self._set_norm(x_tr, y_tr)

        xn_tr = (x_tr - self.in_mean) / self.in_std
        yn_tr = (y_tr - self.out_mean) / self.out_std
        xn_va = (x_va - self.in_mean) / self.in_std
        yn_va = (y_va - self.out_mean) / self.out_std

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Per-member bootstrap: each member samples its own batch indices.
        n = len(xn_tr)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        best, best_state, bad = float("inf"), copy.deepcopy(self.net.state_dict()), 0
        for epoch in range(max_epochs):
            self.net.train()
            for _ in range(steps_per_epoch):
                idx = torch.randint(n, (self.K, batch_size), device=device, generator=g)
                pred = self._raw(xn_tr[idx])  # (K, B, d) with a different batch per member
                loss = ((pred - yn_tr[idx]) ** 2).mean(dim=(1, 2)).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i in range(0, len(xn_va), 8192):
                    p = self._raw(xn_va[i : i + 8192])
                    # mean over dims and members, summed over the chunk
                    val_loss += ((p - yn_va[i : i + 8192]) ** 2).mean(-1).mean(0).sum().item()
                val_loss /= max(len(xn_va), 1)

            if val_loss < best - 1e-6:
                best, best_state, bad = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
            if log_fn is not None and (epoch % 5 == 0 or bad >= patience):
                log_fn(f"  dynamics epoch {epoch}: val={val_loss:.5f} best={best:.5f}")
            if bad >= patience:
                break

        self.net.load_state_dict(best_state)
        self.eval()

        # Calibrate the filter threshold on real held-out transitions.
        with torch.no_grad():
            d = []
            for i in range(0, len(x_va), 8192):
                obs = x_va[i : i + 8192, : self.obs_dim]
                act = x_va[i : i + 8192, self.obs_dim :]
                d.append(self.disagreement_of(self.predict_all(obs, act)))
            d = torch.cat(d)
            self.threshold.fill_(float(torch.quantile(d.float().cpu(), threshold_quantile)))

            mean_pred, _ = self.step(x_va[:, : self.obs_dim], x_va[:, self.obs_dim :])
            target = x_va[:, : self.obs_dim] + y_va
            rmse = ((mean_pred - target) / self.out_std).pow(2).mean().sqrt().item()

        return {"val_loss": best, "val_rmse_norm": rmse, "threshold": float(self.threshold)}

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "K": self.K,
                "hidden_dims": list(self.hidden_dims),
                "state": self.state_dict(),
            },
            tmp,
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DynamicsEnsemble":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["K"], tuple(ckpt["hidden_dims"]))
        model.load_state_dict(ckpt["state"])
        return model.to(device).eval()


def train_dynamics_ensemble(
    dataset: OfflineDataset,
    K: int = 5,
    hidden_dims: Tuple[int, ...] = (256, 256),
    val_frac: float = 0.1,
    seed: int = 0,
    max_epochs: int = 200,
    steps_per_epoch: int = 500,
    log_fn=print,
) -> Tuple[DynamicsEnsemble, Dict[str, float]]:
    """Fit an ensemble on the *real* rows of `dataset` with a random held-out split."""
    real = dataset.real_only()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(real))
    n_val = max(1, int(len(real) * val_frac))
    val, train = real.subset(perm[:n_val]), real.subset(perm[n_val:])

    torch.manual_seed(seed)
    model = DynamicsEnsemble(
        real.data["observations"].shape[1], real.data["actions"].shape[1], K, hidden_dims
    ).to(dataset.device)
    info = model.fit(
        train, val, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch, seed=seed,
        log_fn=log_fn,
    )
    return model, info
