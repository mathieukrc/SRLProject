from typing import Callable, Optional, Union

from data.offline_dataset import OfflineDataset
from FISOR.ensemble_feasibility_critic import EnsembleFeasibilityCritic
from FISOR.feasibility_critic import FeasibilityCritic


def train_feasibility_critic(
    critic: Union[FeasibilityCritic, EnsembleFeasibilityCritic],
    dataset: OfflineDataset,
    num_steps: int,
    batch_size: int = 256,
    log_every: int = 1000,
    log_fn: Optional[Callable[[dict, int], None]] = None,
) -> Union[FeasibilityCritic, EnsembleFeasibilityCritic]:
    for step in range(1, num_steps + 1):
        batch = dataset.sample(batch_size)

        info = critic.update(batch)

        if log_fn is not None and step % log_every == 0:
            log_fn(info, step)

    return critic
