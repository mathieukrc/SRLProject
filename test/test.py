import itertools
from typing import Dict

import numpy as np
import torch

from FISOR.feasibility_critic import FeasibilityCritic

from .evaluation_instruction import EvaluationInstruction


def evaluate_v(
    critic: FeasibilityCritic, instructions: EvaluationInstruction
) -> Dict[tuple, bool]:
    param_values = [
        np.arange(start, end, step) if start != end else [start]
        for start, end, step in zip(
            instructions["start"], instructions["end"], instructions["step"]
        )
    ]

    res = {}

    for params in list(itertools.product(*param_values)):
        v = critic.is_feasible(torch.Tensor(params))
        res[tuple(params)] = v

    return res
