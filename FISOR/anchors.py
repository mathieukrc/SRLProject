"""
Anchor lower bound: the reach-avoid backup implies V_h(s) >= h(s) for every state,
including with the discounted form (1-g)h + g*max(h, V') >= h. Penalising
violations of this bound gives information about states without needing dynamics.

Changes to existing classes (signatures):

  EnsembleFeasibilityCritic.update_vc(
      self, batch: WeightedBatch,
      anchors: Optional[AnchorBatch] = None, anchor_weight: float = 0.0,
  ) -> Dict[str, float]
      loss = expectile loss (as now) + anchor_weight * anchor_lower_bound_loss(
                 self.safe_value(anchors["observations"]), anchors["h"],
                 anchors["bootstrap_weights"])
      log "anchor_loss" and "anchor_violation_rate" = mean(V < h).

  EnsembleFeasibilityCritic.update(
      self, batch, anchors: Optional[AnchorBatch] = None, anchor_weight: float = 0.0)

  FeasibilityCritic: same two changes (weights=None), so FISORAgent keeps working
  with either critic.

  FISORAgent.__init__(self, obs_dim, act_dim, config, device=...,
      feasibility_critic: Optional[Union[FeasibilityCritic, EnsembleFeasibilityCritic]] = None)
      Pass the trained ensemble critic from the augmentation stage instead of
      building a fresh FeasibilityCritic.

  FISORAgent.update(self, batch: MixedBatch,
      anchors: Optional[AnchorBatch] = None, anchor_weight: float = 0.0,
      use_synthetic_actions: bool = False) -> Dict[str, float]
      - reward critic: real rows only (synthetic rewards are zeros)
      - feasibility critic: all rows + anchors
      - actor: real rows, plus synthetic rows if use_synthetic_actions
      - batch must be shuffled before the critic_batch_size slicing
      - bootstrap_weights must be present when the critic is the ensemble
"""

from typing import Optional

import torch


def anchor_lower_bound_loss(
    v: torch.Tensor, h: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    v: (B,) or (K, B) value predictions; h: (B,) training-h at the same states;
    weights: optional (K, B) bootstrap weights.
    Returns sum over members of the (weighted) mean of relu(h - v)^2.
    """
    gap = torch.relu(h - v) ** 2
    if weights is None:
        return gap.mean(dim=-1).sum()
    w = weights if gap.dim() == 2 else weights.mean(dim=0)
    return ((gap * w).sum(-1) / w.sum(-1).clamp_min(1e-8)).sum()
