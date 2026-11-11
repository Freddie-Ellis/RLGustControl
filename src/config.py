from functools import cache
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent


class RigParams(BaseModel):
    """The wing and its rig: the ONE place these are defined. The tunnel env, the mock rig, the CFD case design and
    the tap selection all read them from `CONFIG.rig`. A run's config.json records the copy it ran with."""
    model_config = ConfigDict(extra="forbid")

    n_taps: int = 50
    n_surfaces: int = 12
    # Travel of every control surface. 0 deg is neutral (flush with the wing): the safe state, and where the
    # surfaces sit in a baseline episode. The policy's action -1..1 maps linearly onto min..max.
    actuator_min_deg: float = -50
    actuator_max_deg: float = 50
    # The splitter surfaces only open one way, from 0 deg (closed) up to their own maximum.
    splitter_surfaces: tuple[int, ...] = (8, 9, 10, 11)  # placeholder: which surfaces are splitters
    splitter_max_deg: float = 70  # placeholder
    # Calibration: Cm = My / (q S c).
    q_inf: float = 500.0  # dynamic pressure [Pa], ~29 m/s
    area: float = 0.12  # reference area [m^2]
    mac: float = 0.35  # mean aerodynamic chord [m]

    @model_validator(mode="after")
    def _check_splitters(self) -> Self:
        if bad := [k for k in self.splitter_surfaces if not 0 <= k < self.n_surfaces]:
            raise ValueError(f"splitter_surfaces {bad} are not surfaces 0..{self.n_surfaces - 1}")
        return self

    @property
    def min_deg(self) -> Tensor:
        """Lower stop of each surface [deg], shape (n_surfaces,): splitters stop at 0 (closed)."""
        bottom = torch.full((self.n_surfaces,), self.actuator_min_deg, dtype=torch.float32)
        bottom[list(self.splitter_surfaces)] = 0.0
        return bottom

    @property
    def max_deg(self) -> Tensor:
        """Full opening of each surface [deg], shape (n_surfaces,): splitters have their own maximum."""
        top = torch.full((self.n_surfaces,), self.actuator_max_deg, dtype=torch.float32)
        top[list(self.splitter_surfaces)] = self.splitter_max_deg
        return top

    @property
    def neutral_action(self) -> Tensor:
        """The action that gives 0 deg on each surface, shape (n_surfaces,): -1 for one-sided travel (0..20), 0 for
        symmetric travel (-15..15)."""
        bottom = self.min_deg
        return (-1.0 - 2.0 * bottom / (self.max_deg - bottom)).clamp(-1.0, 1.0)

    def to_deg(self, action: Tensor) -> Tensor:
        """Action in [-1, 1] per surface [..., n_surfaces] -> deflection [deg]."""
        bottom = self.min_deg
        return bottom + 0.5 * (action + 1.0) * (self.max_deg - bottom)

    def deflection_fraction(self, action: Tensor) -> Tensor:
        """How far from neutral, as a fraction of the longest travel either way: 0 at neutral, +-1 at the far stop."""
        neutral = self.neutral_action
        return (action - neutral) / torch.maximum(1.0 - neutral, 1.0 + neutral)


@cache
def _cuda_available() -> bool:
    """Asked once, on first use: the CUDA device query takes over a second, too slow to pay on every import."""
    return torch.cuda.is_available()


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)   # read-only after creation

    # paths
    runs_dir: Path = ROOT / "runs"
    configs_dir: Path = ROOT / "configs"
    profiles_dir: Path = ROOT / "profiles"

    # the wing and its rig
    rig: RigParams = RigParams()

    # compute
    device: Literal["auto", "cpu", "cuda"] = "auto"  # auto = cuda if available

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if _cuda_available() else "cpu")
        return torch.device(self.device)


CONFIG = Config()
