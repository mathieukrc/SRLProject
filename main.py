import matplotlib.pyplot as plt
import numpy as np
import torch

from controlled.env_api import get_offline_observations
from data.offline_dataset import OfflineDataset
from FISOR.feasibility_critic import FeasibilityCritic
from test.evaluation_instruction import EvaluationInstruction
from test.metrics import evaluation_sets_controlled_env, feasibility_metrics
from test.test import evaluate_v
from train.train import train_feasibility_critic

if __name__ == "__main__":
    name = "test_dict"
    dataset = OfflineDataset(get_offline_observations())

    evaluation_sets = evaluation_sets_controlled_env(dataset)

    cont = False
    if cont:
        ckpt = torch.load(name)
        critic = FeasibilityCritic(**ckpt["config"])
        critic.load_state_dict(ckpt)
    else:
        critic = FeasibilityCritic(4, 2, discount=0.6)

    def log_fn(info, i):
        print(f"{i}: vc={info['vc']:.3f} vc_min={info['vc_min']:.3f}")

        for name, states in evaluation_sets.items():
            m = feasibility_metrics(critic, states)

            print(
                f"  {name}: acc={m['accuracy']:.3f} "
                + f"false_feasible={m['false_feasible_rate']:.3f} "
                + f"pred_feas={m['pred_feasible_frac']:.3f}/{m['true_feasible_frac']:.3f}"
            )

    print("Begining training.")
    critic = train_feasibility_critic(
        critic, dataset, 50000, log_every=5000, log_fn=log_fn
    )

    torch.save(critic.state_dict(), name)
    ckpt = torch.load(name)
    critic = FeasibilityCritic(**ckpt["config"])
    critic.load_state_dict(ckpt)

    instructions = EvaluationInstruction(
        start=[-1.0, -1.0, 0.6, 0.0],
        end=[1.0, 1.0, 0.6, 0.0],
        step=[0.01, 0.01, 0.0, 0.0],
    )

    solutions = evaluate_v(critic, instructions)

    vals = np.arange(
        instructions["start"][0], instructions["end"][0], instructions["step"][0]
    )

    arena = np.zeros((len(vals), len(vals), 3))

    for k, v in solutions.items():
        x, y, _, _ = k

        x = np.searchsorted(vals, x)
        y = np.searchsorted(vals, y)

        arena[y, x] = np.array((0, 255, 0)) if v else np.array((255, 0, 0))

    plt.imshow(arena)
    plt.show()
