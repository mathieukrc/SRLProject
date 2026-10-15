from typing import Dict, Optional

import numpy as np
import torch

from .batch import Batch
from .utils import compute_masks, cost_to_h


class OfflineDataset:
    data: Dict[str, torch.Tensor]
    device: torch.device
    _generator: torch.Generator

    def __init__(
        self,
        dataset_dict: Dict[str, np.ndarray],
        cost_scale: float = 25.0,
        clip_eps: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        data_dict = dict(dataset_dict)

        costs = cost_to_h(data_dict["costs"], cost_scale)

        masks = compute_masks(data_dict["terminals"], data_dict["timeouts"])

        lim = 1 - clip_eps
        actions = np.clip(data_dict["actions"], -lim, lim)

        arrays = {
            "observations": data_dict["observations"],
            "actions": actions,
            "next_observations": data_dict["next_observations"],
            "costs": costs.reshape(-1),
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

        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)

    def __len__(self) -> int:
        return self.data["observations"].shape[0]

    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(
            high=len(self),
            size=(batch_size,),
            device=self.device,
            generator=self._generator,
        )
        return Batch(**{k: v[idx] for k, v in self.data.items()})
