"""Per-experiment configuration. One `RunConfig` fully describes how a run was built and trained.

Every run trains a `SquashedGaussianPolicy` with SAC; only the environment varies. Each env config has a `kind` tag
that the registry uses to pick the env class.
"""

import json
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.envs.base import EnvSpec
from src.ml.envs.gust_lift_dummy import GustLiftParams
from src.ml.envs.wind_tunnel import WindTunnelParams
from src.ml.envs.xfoil_pitch import XFOILPitchProblemParams


class StrictModel(BaseModel):
    """Rejects unknown keys, so a typo in a config file is an error instead of a silent default."""
    model_config = ConfigDict(extra="forbid")


# ----- environments -----
class GustLiftEnvConfig(StrictModel):
    kind: Literal["gust_lift_dummy"] = "gust_lift_dummy"
    spec: EnvSpec = EnvSpec()
    params: GustLiftParams = GustLiftParams()


class XFOILPitchEnvConfig(StrictModel):
    kind: Literal["xfoil_pitch"] = "xfoil_pitch"
    spec: EnvSpec = EnvSpec()
    params: XFOILPitchProblemParams = XFOILPitchProblemParams()


class WindTunnelEnvConfig(StrictModel):
    kind: Literal["wind_tunnel"] = "wind_tunnel"
    spec: EnvSpec = EnvSpec()  # spec.dt is the real control period in the tunnel
    params: WindTunnelParams = WindTunnelParams()


type EnvConfig = Annotated[
    GustLiftEnvConfig | XFOILPitchEnvConfig | WindTunnelEnvConfig, Field(discriminator="kind")
]


# ----- policy -----
class PolicyConfig(StrictModel):
    hidden: int = 256
    init_log_std: float = -0.5


# ----- SAC -----
class SACConfig(StrictModel):
    iterations: int = 100  # one iteration = one episode in every parallel env, then a metrics row
    n_envs: int = 16  # parallel sim envs (always 1 in the tunnel)
    gamma: float = 0.99  # discount; horizon of ~1 / (1 - gamma) = 100 control steps
    tau: float = 0.005  # target critic Polyak rate
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_transitions: int = 5_000  # uniform random actions before the policy takes over, let the critic see the whole action range -- TODO: High enough for 12 inputs?
    updates_per_step: int = 1  # gradient steps per env step; raise for scarce tunnel data (with critic dropout)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.1  # entropy weight at the start
    target_entropy: float | None = None  # None -> -act_dim
    critic_hidden: int = 256
    critic_dropout: float = 0.0  # DroQ uses ~0.01 with updates_per_step ~ 20
    critic_layer_norm: bool = True
    critic_privileged: bool = True  # give the critic env.privileged() (e.g. balance loads); False = same view as actor


# ----- decoupled actor / learner -----
class DistributedConfig(StrictModel):
    """Settings for the split setup (actor on the tunnel PC, learner on the laptop). Ignored by `train.py`.

    The learning rate of data use is still `SACConfig.updates_per_step`: the learner does that many updates per
    env step it receives, so it learns as fast as data arrives and does not overfit while the tunnel resets.
    """
    publish_every: int = 250  # SAC updates between published weight versions
    eval_every: int = 10  # every k-th actor episode runs the deterministic policy (no exploration)


# ----- run -----
class RunConfig(StrictModel):
    name: str = "run"
    seed: int = 0
    env: EnvConfig = GustLiftEnvConfig()
    policy: PolicyConfig = PolicyConfig()
    algo: SACConfig = SACConfig()
    distributed: DistributedConfig = DistributedConfig()

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_kinds(cls, data: Any) -> Any:
        """Configs saved before the project went SAC-only tag `policy` and `algo` with a `kind`. Drop the tags so
        old SAC runs still load; old PPO runs fail here with a clear message."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for section, expected in (("policy", "squashed_gaussian"), ("algo", "sac")):
            if isinstance(part := data.get(section), dict) and "kind" in part:
                kind = part["kind"]
                if kind != expected:
                    raise ValueError(f"{section} kind {kind!r} is no longer supported; every run uses {expected!r}")
                data[section] = {k: v for k, v in part.items() if k != "kind"}
        return data

    @classmethod
    def from_toml(cls, path: Path) -> RunConfig:
        with open(path, "rb") as f:
            return cls.model_validate(tomllib.load(f))

    @classmethod
    def from_toml_str(cls, text: str) -> RunConfig:
        return cls.model_validate(tomllib.loads(text))

    def to_toml(self) -> str:
        return _to_toml(self.model_dump(mode="json"))


type _TomlValue = str | int | float | bool | list[_TomlValue] | dict[str, _TomlValue] | None


def _toml_scalar(v: _TomlValue) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # JSON string escapes are valid TOML basic strings
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} as a TOML value")


def _to_toml(data: dict[str, _TomlValue], prefix: str = "") -> str:
    """Minimal TOML writer for nested dicts of scalars/lists (stdlib tomllib can only read). None values are skipped."""
    lines: list[str] = []
    tables: list[tuple[str, dict[str, _TomlValue]]] = []
    for k, v in data.items():
        if isinstance(v, dict):
            tables.append((f"{prefix}{k}", v))
        elif v is not None:
            lines.append(f"{k} = {_toml_scalar(v)}")
    out = "\n".join(lines)
    for name, table in tables:
        body = _to_toml(table, f"{name}.")
        out += f"\n\n[{name}]\n{body}" if out or prefix else f"[{name}]\n{body}"
    return out.strip("\n") + ("\n" if not prefix else "")
