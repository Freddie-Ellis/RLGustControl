import torch

from src.ml.envs.base import EnvSpec
from src.ml.envs.gust_lift_dummy import GustLiftDummyEnv, GustLiftParams

if __name__ == "__main__":
    # Example usage of the GustLiftDummyEnv
    spec = EnvSpec()

    env = GustLiftDummyEnv(spec=spec, params=GustLiftParams(), n_envs=1, seed=42)
    obs = env.reset()
    print(f"Initial observation: {obs}")

    action = torch.tensor([[0.5]])  # Example action
    next_obs, reward = env.step(action)
    print(f"Next observation: {next_obs}, Reward: {reward}")