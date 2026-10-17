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
        }

        n = len(arrays["observations"])
        for k, v in arrays.items():
            if len(v) != n:
                raise ValueError(f"'{k}' has length {len(v)}, expected {n}")

        self.data = {
            k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
            for k, v in arrays.items()
        }

        self.set_bootstrap_weights(n_members)

        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)

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
        if num_members == 1:
            self.data["bootstrap_weights"] = torch.ones((1, len(self))).to(self.device)
        else:
            self.data["bootstrap_weights"] = torch.poisson(
                torch.ones((num_members, len(self)))
            ).to(self.device)

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

    def subset(self, index: np.ndarray):
        sub = copy.deepcopy(self)
        index_tensor = torch.tensor(index, device=self.device)

        sub.data = {
            k: (v[:, index_tensor] if k == "bootstrap_weights" else v[index_tensor])
            for k, v in self.data.items()
        }

        return sub
