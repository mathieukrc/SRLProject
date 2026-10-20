"""
One ExperimentConfig = one run. A run is a pure function of its config, which is what
makes the cluster setup simple: run ids are hashes of the config, results never collide,
and resubmitting the same manifest skips everything that already finished.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # ---- what is being compared
    env: str = "controlled"
    variant: str = "coverage"  # "coverage" or "size_matched"
    coverage: float = 100.0  # % of boundary-approaching trajectories kept
    rule: str = "none"  # see generation.generator.RULES
    budget_frac: float = 0.0  # total synthetic transitions / variant size, over all rounds
    label_noise: float = 0.0  # std of noise on synthetic h labels (h units)
    dynamics: str = "auto"  # "true", "learned" or "auto" (true when the env has it); "none" for rule none
    seed: int = 0

    # ---- feasibility critic (FISOR)
    K: int = 5  # bootstrap ensemble size
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    discount: float = 0.99
    expectile: float = 0.9
    critic_lr: float = 3e-4
    batch_size: int = 256
    h_pos_scale: float = 1.0  # shape_h: scale of violations
    h_neg_scale: float = 1.0  # shape_h: scale of safe margins

    # ---- schedule (identical for every rule, including "none")
    round0_steps: int = 50_000
    rounds: int = 3
    steps_per_round: int = 10_000
    reset_critic_each_round: bool = False  # False: fine-tune from previous round

    # ---- generation
    uncertainty: str = "entropy"  # "entropy" or "std"
    beta: Optional[float] = 1.0  # None: top-k; float: sample proportional to score**beta
    candidate_multiplier: int = 10
    noise_scale: float = 0.05
    branching: int = 4
    chain_len: int = 5
    n_candidates: int = 32
    lookahead: int = 5
    nn_noise: float = 0.1
    policy_noise: float = 0.1
    disagreement_scale: float = 1.0  # multiplies the calibrated dynamics threshold

    # ---- dynamics ensemble (only when dynamics == learned)
    dyn_K: int = 5
    dyn_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    dyn_max_epochs: int = 200
    dyn_steps_per_epoch: int = 500

    # ---- policy stage (FISOR actor + reward critic trained on real data only)
    policy: bool = False
    policy_steps: int = 100_000
    policy_batch_size: int = 2048
    eval_episodes: int = 10

    # ---- bookkeeping
    log_every: int = 5_000
    tag: str = ""

    # fields that do not change the result: excluded from the run id
    _NON_IDENTITY = ("log_every", "tag")

    def normalised(self) -> "ExperimentConfig":
        """Canonical form, so equivalent configs get the same run id."""
        c = dataclasses.replace(self)
        if c.dynamics not in ("true", "learned", "auto", "none"):
            raise ValueError(f"dynamics must be 'true', 'learned' or 'auto' (quote it in YAML), got {c.dynamics!r}")
        if c.variant not in ("coverage", "size_matched"):
            raise ValueError(f"unknown variant {c.variant!r}")
        from generation.generator import RULES

        if c.rule not in RULES:
            raise ValueError(f"unknown rule {c.rule!r}; choose from {RULES}")
        if c.rule == "none":
            # no synthetic data: budget, label noise and dynamics model are irrelevant,
            # so all baselines of one variant/seed collapse into a single run
            c.budget_frac, c.label_noise, c.dynamics = 0.0, 0.0, "none"
        if c.variant == "size_matched" and c.coverage >= 100:
            c.variant = "coverage"  # identical datasets
        if c.dynamics == "auto" and c.rule != "none":
            from experiments.envs import ENVS

            c.dynamics = "true" if ENVS[c.env].true_step is not None else "learned"
        c.coverage = float(c.coverage)
        return c

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    # ---------------------------------------------------------------- ids
    def _hash(self, keys) -> str:
        d = self.to_dict()
        payload = json.dumps({k: d[k] for k in sorted(keys)}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    @property
    def run_id(self) -> str:
        keys = [k for k in self.to_dict() if k not in self._NON_IDENTITY]
        readable = f"{self.env}_{self.variant[:4]}{int(self.coverage)}_{self.rule}_s{self.seed}"
        return f"{readable}_{self._hash(keys)}"

    # Everything that determines the dataset variant.
    VARIANT_KEYS = ("env", "variant", "coverage", "seed")
    # ... plus everything that determines the round-0 critic (shared by every rule).
    ROUND0_KEYS = VARIANT_KEYS + (
        "K", "hidden_dims", "discount", "expectile", "critic_lr", "batch_size",
        "h_pos_scale", "h_neg_scale", "round0_steps",
    )
    DYNAMICS_KEYS = VARIANT_KEYS + ("dyn_K", "dyn_hidden_dims", "dyn_max_epochs", "dyn_steps_per_epoch")

    @property
    def variant_key(self) -> str:
        return f"{self.env}_{self.variant[:4]}{int(self.coverage)}_s{self.seed}_{self._hash(self.VARIANT_KEYS)}"

    @property
    def round0_key(self) -> str:
        return f"{self.env}_{self.variant[:4]}{int(self.coverage)}_s{self.seed}_{self._hash(self.ROUND0_KEYS)}"

    @property
    def dynamics_key(self) -> str:
        return f"{self.env}_{self.variant[:4]}{int(self.coverage)}_s{self.seed}_{self._hash(self.DYNAMICS_KEYS)}"
