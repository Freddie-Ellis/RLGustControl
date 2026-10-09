"""Turns config models into objects. The only place that maps a `kind` to a class."""

from typing import Any

from src.ml.algos.base import Algorithm
from src.ml.algos.ppo import PPO
from src.ml.algos.sac import SAC
from src.ml.envs.base import Env
from src.ml.envs.gust_lift_dummy import GustLiftDummyEnv
from src.ml.envs.xfoil_pitch import XFOILPitchProblemEnv
from src.ml.policies.base import Policy
from src.ml.policies.gaussian import GaussianPolicy
from src.ml.policies.squashed_gaussian import SquashedGaussianPolicy
from src.ml.run_config import AlgoConfig, EnvConfig, PolicyConfig


def build_env(cfg: EnvConfig, n_envs: int, seed: int | None = None) -> Env:
    match cfg.kind:
        case "gust_lift_dummy":
            return GustLiftDummyEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
        case "xfoil_pitch":
            return XFOILPitchProblemEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)

def build_policy(cfg: PolicyConfig, obs_dim: int, act_dim: int) -> Policy:
    match cfg.kind:
        case "gaussian_mlp":
            return GaussianPolicy(obs_dim, act_dim, hidden=cfg.hidden, init_log_std=cfg.init_log_std)
        case "squashed_gaussian":
            return SquashedGaussianPolicy(obs_dim, act_dim, hidden=cfg.hidden, init_log_std=cfg.init_log_std)


def build_algo(cfg: AlgoConfig) -> Algorithm[Any]:
    """Typed loosely on purpose: `RunConfig` has already checked the algorithm suits the policy."""
    match cfg.kind:
        case "ppo":
            return PPO(cfg)
        case "sac":
            return SAC(cfg)
