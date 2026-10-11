import dsrl
import gymnasium as gym


def list_envs() -> None:
    print(sorted(k for k in gym.envs.registry if k.startswith("Offline")))


def load_env(name: str) -> None:
    env = gym.make(name)
    dataset = env.get_dataset()

    for k, v in dataset.items():
        print(k, v.shape)


if __name__ == "__main__":
    offline_point_goal = "OfflinePointGoal1-v0"
    load_env(offline_point_goal)
