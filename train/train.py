from typing import Callable, Optional

from data.offline_dataset import OfflineDataset
from FISOR.feasibility_critic import FeasibilityCritic


def train_feasibility_critic(
    critic: FeasibilityCritic,
    dataset: OfflineDataset,
    num_steps: int,
    batch_size: int = 256,
    log_every: int = 1000,
    log_fn: Optional[Callable[[dict, int], None]] = None,
) -> FeasibilityCritic:
    for step in range(1, num_steps + 1):
        batch = dataset.sample(batch_size)

        info = critic.update(batch)

        if log_fn is not None and step % log_every == 0:
            log_fn(info, step)

    return critic
