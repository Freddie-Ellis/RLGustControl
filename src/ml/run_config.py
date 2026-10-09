"""Per-experiment configuration. One `RunConfig` fully describes how a run was built and trained.

Each component config has a `kind` tag that the registry uses to pick the class. To add a second
env/policy/algo, write its config model and turn the alias below into a tagged union, e.g.
    type EnvConfig = Annotated[GustLiftEnvConfig | WingEnvConfig, Field(discriminator="kind")]
"""

import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.envs.base import EnvSpec
from src.ml.envs.gust_lift_dummy import GustLiftParams
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


type EnvConfig = Annotated[GustLiftEnvConfig | XFOILPitchEnvConfig, Field(discriminator="kind")]


# ----- policies -----
class GaussianMLPConfig(StrictModel):
    kind: Literal["gaussian_mlp"] = "gaussian_mlp"
    hidden: int = 64
    init_log_std: float = -0.5


class SquashedGaussianConfig(StrictModel):
    kind: Literal["squashed_gaussian"] = "squashed_gaussian"
    hidden: int = 256
    init_log_std: float = -0.5


type PolicyConfig = Annotated[GaussianMLPConfig | SquashedGaussianConfig, Field(discriminator="kind")]


# ----- algorithms -----
class AlgoConfigBase(StrictModel):
    """Fields every algorithm has. `train.py` and the UI only rely on these, so they work for any algorithm."""
    iterations: int = 60  # one iteration = one episode in every parallel env, then a metrics row
    n_envs: int = 64  # parallel sim envs (always 1 in the tunnel)


class PPOConfig(AlgoConfigBase):
    kind: Literal["ppo"] = "ppo"
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 8
    minibatch: int = 1024
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class SACConfig(AlgoConfigBase):
    kind: Literal["sac"] = "sac"
    iterations: int = 100
    n_envs: int = 16
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


type AlgoConfig = Annotated[PPOConfig | SACConfig, Field(discriminator="kind")]

# Which policy kinds each algorithm can train (mirrors PPOPolicy / SACPolicy in policies/base.py).
COMPATIBLE_POLICIES: dict[str, set[str]] = {
    "ppo": {"gaussian_mlp"},
    "sac": {"squashed_gaussian"},
}


# ----- run -----
class RunConfig(StrictModel):
    name: str = "run"
    seed: int = 0
    env: EnvConfig = GustLiftEnvConfig()
    policy: PolicyConfig = GaussianMLPConfig()
    algo: AlgoConfig = PPOConfig()

    @model_validator(mode="after")
    def _check_policy_suits_algo(self) -> RunConfig:
        """Fail when the config is loaded, not after the tunnel is spun up and the first update crashes."""
        allowed = COMPATIBLE_POLICIES[self.algo.kind]
        if self.policy.kind not in allowed:
            raise ValueError(
                f"algo {self.algo.kind!r} cannot train policy {self.policy.kind!r}; use one of {sorted(allowed)}"
            )
        return self

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
