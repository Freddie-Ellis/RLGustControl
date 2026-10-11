"""The boundary between the code and the wind tunnel rig.

`WindTunnelEnv` only talks to the rig through `TunnelIO`, so the same env runs against `MockTunnelIO` on a laptop
and against the real DAQ and servos in the tunnel. Units are physical here (Pa, N, N m, degrees); the env turns them
into normalised observations and Cm.
"""

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True)
class SensorBlock:
    """Every DAQ sample that arrived since the last read. M samples (M varies with timing; may be 0)."""
    t: Tensor  # [M] seconds since the IO was opened
    taps: Tensor  # [M, n_taps] surface pressures [Pa]
    loads: Tensor  # [M, 3] balance: Fx, Fz [N], My [N m] about the moment reference point


class TunnelIO(Protocol):
    """What the env needs from the rig. The DAQ streams continuously in the background (in hardware or a driver
    thread); `read_block` just collects what has arrived. That is what lets a fast, never-stopping tunnel feed a
    step-based env."""

    n_taps: int
    n_surfaces: int

    def read_block(self) -> SensorBlock:
        """All samples since the previous call."""
        ...

    def write_surfaces(self, deg: Tensor) -> None:
        """Command every control surface [n_surfaces], in degrees. Takes effect after the actuator delay."""
        ...

    def set_turbulence(self, level: float) -> None:
        """Set the turbulence device (here: rms gust angle in degrees)."""
        ...

    def safe_state(self) -> None:
        """All surfaces to neutral. Must always work, whatever state the code is in."""
        ...

    def close(self) -> None:
        ...
