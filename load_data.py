import dsrl
import gymnasium as gym

from constants import OFFLINE_CAR_CIRCLE, OFFLINE_GOAL_POINT


def list_envs() -> None:
    print(sorted(k for k in gym.envs.registry if k.startswith("Offline")))


def load_env(name: str) -> None:
    env = gym.make(name)
    dataset = env.get_dataset()

    for k, v in dataset.items():
        print(k, v.shape)


if __name__ == "__main__":
    print("Loading datasets into memory (and downloading if missing)")
    list_envs()
    print("Loading " + OFFLINE_CAR_CIRCLE + "...")
    load_env(OFFLINE_CAR_CIRCLE)
    print("Loading " + OFFLINE_GOAL_POINT + "...")
    load_env(OFFLINE_GOAL_POINT)
