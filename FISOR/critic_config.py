from typing import List, TypedDict


class CriticConfig(TypedDict):
    obs_dim: int
    act_dim: int
    hidden_dims: List[int]
    num_qs: int
    discount: float
    tau: float
    expectile: float
    critic_lr: float
    value_lr: float
    critic_batch_size: int


class EnsembleCriticConfig(CriticConfig):
    K: int
