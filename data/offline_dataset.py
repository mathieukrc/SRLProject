import copy
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .batch import WeightedBatch
from .utils import HFn, compute_masks, cost_to_h


class OfflineDataset:
    data: Dict[str, torch.Tensor]
    device: torch.device
    h_fn: Optional[HFn]
    _generator: torch.Generator

    def __init__(
        self,
        dataset_dict: Dict[str, np.ndarray],
        cost_scale: float = 25.0,
        clip_eps: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_members: int = 1,
        seed: Optional[int] = None,
        timeout_is_done: bool = True,
        h_fn: Optional[HFn] = None,
    ) -> None:
        self.device = torch.device(device)
        self.h_fn = h_fn
        self.cost_scale = cost_scale
        self.clip_eps = clip_eps
        self.timeout_is_done = timeout_is_done
        self.seed = seed
        data_dict = dict(dataset_dict)

        masks = compute_masks(
            data_dict["terminals"],
            data_dict["timeouts"],
            timeout_is_done=timeout_is_done,
        )

        if h_fn is None:
            costs = cost_to_h(data_dict["costs"], cost_scale)
        else:
            costs = self._dense_costs(
                h_fn, data_dict["observations"], data_dict["next_observations"], masks
            )

        lim = 1 - clip_eps
        actions = np.clip(data_dict["actions"], -lim, lim)

        arrays = {
            "observations": data_dict["observations"],
            "actions": actions,
            "next_observations": data_dict["next_observations"],
            "costs": np.asarray(costs).reshape(-1),
            "rewards": data_dict["rewards"],
            "masks": masks.reshape(-1),
            "is_synthetic": np.asarray(
                data_dict.get("is_synthetic", np.zeros(len(masks))), dtype=np.float32
            ).reshape(-1),
        }

        n = len(arrays["observations"])
        for k, v in arrays.items():
            if len(v) != n:
                raise ValueError(f"'{k}' has length {len(v)}, expected {n}")

        self.data = {
            k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
            for k, v in arrays.items()
        }

        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)

        self.set_bootstrap_weights(n_members)

    @staticmethod
    @torch.no_grad()
    def _dense_costs(
        h_fn: HFn,
        observations: np.ndarray,
        next_observations: np.ndarray,
        masks: np.ndarray,
        chunk: int = 65536,
    ) -> np.ndarray:
        def h_of(states: np.ndarray) -> torch.Tensor:
            out = [
                h_fn(torch.as_tensor(states[i : i + chunk], dtype=torch.float32))
                .reshape(-1)
                .cpu()
                for i in range(0, len(states), chunk)
            ]
            h = torch.cat(out)
            if len(h) != len(states):
                raise ValueError(
                    f"h_fn returned {len(h)} values for {len(states)} states; "
                    "it must return one value per state"
                )
            if not torch.isfinite(h).all():
                raise ValueError("h_fn returned non-finite values")
            return h

        h_s = h_of(np.asarray(observations))
        h_next = h_of(np.asarray(next_observations))
        terminal = torch.as_tensor(np.asarray(masks) == 0)

        return torch.where(terminal, torch.maximum(h_s, h_next), h_s).numpy()

    def set_bootstrap_weights(self, num_members: int) -> None:
        self.data["bootstrap_weights"] = self._draw_bootstrap_weights(
            num_members, len(self)
        )

    def _draw_bootstrap_weights(self, num_members: int, n: int) -> torch.Tensor:
        if num_members == 1:
            return torch.ones((1, n), device=self.device)
        # Drawn on the dataset's own generator so the weights are reproducible
        # from the seed alone (not from whatever the global RNG state happens to be).
        rates = torch.ones((num_members, n), device=self.device)
        return torch.poisson(rates, generator=self._generator)

    @property
    def n_members(self) -> int:
        return int(self.data["bootstrap_weights"].shape[0])

    @property
    def n_synthetic(self) -> int:
        return int(self.data["is_synthetic"].sum().item())

    def __len__(self) -> int:
        return self.data["observations"].shape[0]

    def sample(self, batch_size: int) -> WeightedBatch:
        idx = torch.randint(
            high=len(self),
            size=(batch_size,),
            device=self.device,
            generator=self._generator,
        )
        return WeightedBatch(
            **{
                k: (v[:, idx] if k == "bootstrap_weights" else v[idx])
                for k, v in self.data.items()
            }
        )

    def subset(self, index: np.ndarray) -> "OfflineDataset":
        index_tensor = torch.as_tensor(np.asarray(index), device=self.device).long()

        # Shallow-copy the object and only slice the tensors: a deepcopy would first
        # duplicate the full dataset in memory, which matters for the DSRL datasets.
        sub = copy.copy(self)
        sub.data = {
            k: (v[:, index_tensor] if k == "bootstrap_weights" else v[index_tensor])
            for k, v in self.data.items()
        }
        sub._generator = torch.Generator(device=self.device)
        sub._generator.set_state(self._generator.get_state())

        return sub

    def real_only(self) -> "OfflineDataset":
        """The rows that came from the offline dataset (no synthetic transitions)."""
        keep = (self.data["is_synthetic"] == 0).nonzero().reshape(-1)
        return self.subset(keep.cpu().numpy())

    def with_synthetic(
        self, raw: Dict[str, np.ndarray], label_noise: float = 0.0
    ) -> "OfflineDataset":
        """
        Returns a new dataset with the synthetic transitions in `raw` appended.

        `raw` uses the same keys as the offline data ('observations', 'actions',
        'next_observations', 'rewards', 'costs', 'terminals', 'timeouts'). It is passed
        through the same constructor path as the real data, so h is computed by the same
        h_fn and terminal handling. Synthetic chains should set timeouts = 0 at the
        chain end: that is a truncation, so the critic must bootstrap from V_h(s_k).

        label_noise: std of Gaussian noise added to the synthetic h labels (in h units),
            for the label-noise experiment. Real rows are never perturbed.
        """
        n_new = len(raw["observations"])
        if n_new == 0:
            return self.subset(np.arange(len(self)))

        raw = dict(raw)
        raw["is_synthetic"] = np.ones(n_new, dtype=np.float32)
        new = OfflineDataset(
            raw,
            cost_scale=self.cost_scale,
            clip_eps=self.clip_eps,
            device=str(self.device),
            n_members=1,
            timeout_is_done=self.timeout_is_done,
            h_fn=self.h_fn,
        )
        new_data = dict(new.data)
        if label_noise > 0:
            noise = torch.randn(
                new_data["costs"].shape, device=self.device, generator=self._generator
            )
            new_data["costs"] = new_data["costs"] + label_noise * noise
        new_data["bootstrap_weights"] = self._draw_bootstrap_weights(
            self.n_members, n_new
        )

        merged = copy.copy(self)
        merged.data = {
            k: torch.cat((v, new_data[k]), dim=1 if k == "bootstrap_weights" else 0)
            for k, v in self.data.items()
        }
        return merged
