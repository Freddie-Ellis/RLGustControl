"""Turns env configs into envs. The only place that maps an env `kind` to a class."""

from src.ml.envs.base import Env, HistoryWrapper
from src.ml.envs.gust_lift_dummy import GustLiftDummyEnv
from src.ml.envs.wind_tunnel import WindTunnelEnv
from src.ml.envs.xfoil_pitch import XFOILPitchProblemEnv
from src.ml.run_config import EnvConfig


def _build(cfg: EnvConfig, n_envs: int, seed: int | None) -> Env:
    match cfg.kind:
        case "gust_lift_dummy":
            return GustLiftDummyEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
        case "xfoil_pitch":
            return XFOILPitchProblemEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)
        case "wind_tunnel":
            return WindTunnelEnv(spec=cfg.spec, params=cfg.params, n_envs=n_envs, seed=seed)


def build_env(cfg: EnvConfig, n_envs: int, seed: int | None = None) -> Env:
    """The env of `cfg`, with the observation history its spec asks for."""
    env = _build(cfg, n_envs, seed)
    return HistoryWrapper(env, cfg.spec.history_len) if cfg.spec.history_len > 1 else env
