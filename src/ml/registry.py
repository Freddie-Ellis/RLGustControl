"""Turns env configs into envs. The only place that maps an env `kind` to a class."""

from src.ml.envs.base import Env
from src.ml.envs.gust_lift_dummy import GustLiftDummyEnv
from src.ml.envs.wind_tunnel import WindTunnelEnv
from src.ml.envs.xfoil_pitch import XFOILPitchProblemEnv
from src.ml.run_config import EnvConfig


def build_env(cfg: EnvConfig, n_envs: int, seed: int | None = None) -> Env:
    match cfg.kind:
        case "gust_lift_dummy":
            return GustLiftDummyEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
        case "xfoil_pitch":
            return XFOILPitchProblemEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
        case "wind_tunnel":
            return WindTunnelEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
