"""Per-experiment configuration. One `RunConfig` fully describes how a run was built and trained.

Each component config has a `kind` tag that the registry uses to pick the class. To add a second
env/policy/algo, write its config model and turn the alias below into a tagged union, e.g.
    type EnvConfig = Annotated[GustLiftEnvConfig | WingEnvConfig, Field(discriminator="kind")]
"""

import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.ml.envs.base import EnvSpec
from src.ml.envs.gust_lift_dummy import GustLiftParams


class StrictModel(BaseModel):
    """Rejects unknown keys, so a typo in a config file is an error instead of a silent default."""
    model_config = ConfigDict(extra="forbid")


# ----- environments -----
class GustLiftEnvConfig(StrictModel):
    kind: Literal["gust_lift_dummy"] = "gust_lift_dummy"
    spec: EnvSpec = EnvSpec()
    params: GustLiftParams = GustLiftParams()


type EnvConfig = GustLiftEnvConfig


# ----- policies -----
class GaussianMLPConfig(StrictModel):
    kind: Literal["gaussian_mlp"] = "gaussian_mlp"
    hidden: int = 64
    init_log_std: float = -0.5


type PolicyConfig = GaussianMLPConfig


# ----- algorithms -----
class PPOConfig(StrictModel):
    kind: Literal["ppo"] = "ppo"
    iterations: int = 60
    n_envs: int = 64
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 8
    minibatch: int = 1024
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


type AlgoConfig = PPOConfig


# ----- run -----
class RunConfig(StrictModel):
    name: str = "run"
    seed: int = 0
    env: EnvConfig = GustLiftEnvConfig()
    policy: PolicyConfig = GaussianMLPConfig()
    algo: AlgoConfig = PPOConfig()

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
