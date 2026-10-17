"""
One ExperimentConfig = one run = one line in a manifest.

Pure Python (no torch import) so manifests can be generated on a login node.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

ENVS = ("controlled", "car_circle", "point_goal")
VARIANTS = ("coverage", "size_matched")
RULES = (
    "none",  # no synthetic data, but the same extra training steps (control)
    "random",
    "density_matched",
    "noise_augmentation",
    "uncertainty_guided",
    "violation_guided",
    "combined",
)
ADAPTIVE_RULES = ("uncertainty_guided", "violation_guided", "combined")
TRANSITION_MODELS = ("learned", "simulator")  # simulator = oracle, controlled env only
ACTION_PROPOSERS = ("mixed", "model_recovery", "behaviour", "random")

# Fields that determine the shared round-0 artefacts (dataset variant, dynamics
# ensemble, baseline critic). All runs with equal values reuse one baseline, so
# every targeting rule starts from the identical critic (paired comparison).
BASELINE_FIELDS = (
    "env",
    "seed",
    "variant",
    "coverage_percent",
    "dense_h",
    "h_pos_scale",
    "h_neg_scale",
    "ensemble_size",
    "discount",
    "expectile",
    "base_steps",
)

# Bookkeeping fields excluded from the run hash.
NON_IDENTITY_FIELDS = ("tag",)


@dataclass(frozen=True)
class ExperimentConfig:
    env: str
    seed: int = 0

    # --- dataset -------------------------------------------------------------
    # size_matched: random whole trajectories removed until the dataset has the
    # same size as the coverage variant with the same coverage_percent.
    variant: str = "coverage"
    coverage_percent: float = 100.0
    dense_h: bool = True  # False -> binary h from sign(h): the dense-h ablation
    h_pos_scale: float = 25.0
    h_neg_scale: float = 1.0

    # --- synthetic generation ------------------------------------------------
    rule: str = "none"
    budget: int = 0  # TOTAL accepted synthetic transitions over all rounds
    n_rounds: int = 3
    horizon: int = 5  # chain length per start state
    action_proposer: str = "mixed"
    transition_model: str = "learned"
    # filter threshold, as a multiple of the 95th-percentile ensemble disagreement on
    # real validation transitions (so it means the same thing in every env)
    max_disagreement: float = 2.0
    beta: Optional[float] = 1.0  # None -> top-k selection in the pickers
    candidate_multiplier: int = 10
    use_transitions: bool = True
    use_anchors: bool = True
    n_anchors: int = 0  # TOTAL anchor states; 0 -> equal to budget
    anchor_weight: float = 1.0
    synthetic_ratio: float = 0.25  # share of each critic batch that is synthetic

    # --- feasibility critic --------------------------------------------------
    ensemble_size: int = 5
    discount: float = 0.99
    expectile: float = 0.9
    base_steps: int = 100_000  # round-0 training (shared baseline)
    round_steps: int = 20_000  # extra steps per round, ALSO for rule "none"
    retrain_each_round: bool = False  # False: fine-tune from the previous round

    # --- policy stage (DSRL) -------------------------------------------------
    train_agent: bool = False
    agent_steps: int = 300_000
    agent_uses_synthetic_actions: bool = False
    eval_episodes: int = 20

    tag: str = ""  # grid block name, used for grouping in the analysis

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"Invalid config ({msg}): {self.to_dict()}")

        check(self.env in ENVS, f"env must be one of {ENVS}")
        check(self.variant in VARIANTS, f"variant must be one of {VARIANTS}")
        check(self.rule in RULES, f"rule must be one of {RULES}")
        check(self.transition_model in TRANSITION_MODELS, "transition_model")
        check(self.action_proposer in ACTION_PROPOSERS, "action_proposer")
        check(0.0 <= self.coverage_percent <= 100.0, "coverage_percent in [0, 100]")
        check((self.rule == "none") == (self.budget == 0), "budget == 0 iff rule == 'none'")
        check(self.n_rounds >= 1 and self.horizon >= 1, "n_rounds, horizon >= 1")
        check(
            self.rule == "none" or self.use_transitions or self.use_anchors,
            "need transitions and/or anchors",
        )
        check(
            self.transition_model == "learned" or self.env == "controlled",
            "simulator transitions only exist for the controlled env",
        )

    # ------------------------------------------------------------------------
    @property
    def total_transitions(self) -> int:
        return self.budget if (self.use_transitions and self.rule != "none") else 0

    @property
    def total_anchors(self) -> int:
        if not self.use_anchors or self.rule == "none":
            return 0
        return self.n_anchors if self.n_anchors > 0 else self.budget

    def per_round(self, total: int, round_idx: int) -> int:
        """Split a total budget over rounds (remainder goes to the first rounds)."""
        base, rem = divmod(total, self.n_rounds)
        return base + (1 if round_idx < rem else 0)

    @property
    def resource(self) -> str:
        """Which manifest / partition this run belongs to."""
        return "gpu" if (self.train_agent or self.env != "controlled") else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def _hash(self, fields) -> str:
        payload = {k: getattr(self, k) for k in fields}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    @property
    def run_id(self) -> str:
        fields = [
            f.name for f in dataclasses.fields(self) if f.name not in NON_IDENTITY_FIELDS
        ]
        return (
            f"{self.env}-{self.variant[:3]}{int(self.coverage_percent)}-{self.rule}"
            f"-b{self.budget}-s{self.seed}-{self._hash(fields)}"
        )

    @property
    def baseline_key(self) -> str:
        return (
            f"{self.env}-{self.variant[:3]}{int(self.coverage_percent)}-s{self.seed}"
            f"-{self._hash(BASELINE_FIELDS)}"
        )

    def baseline_config(self) -> "ExperimentConfig":
        """The config representing this run's shared round-0 artefacts."""
        return dataclasses.replace(
            self, rule="none", budget=0, train_agent=False, tag="baseline"
        )
