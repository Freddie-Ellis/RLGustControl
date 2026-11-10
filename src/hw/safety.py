"""Sits between the policy and the servos. Assumes nothing about the policy: a half-trained network can output
anything, including NaN, full deflection on all surfaces, or a sign flip every step.

The real E-stop is the tunnel's hardware stop. This layer is the software line of defence in front of it.
"""

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class SafetyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rate: float = 20.0  # max change of a normalised command per second (full travel -1 -> 1 in 0.1 s)
    cm_trip: float = 0.3  # |Cm| above this trips: surfaces to neutral until the next reset


class SafetyLayer:
    def __init__(self, params: SafetyParams, n_surfaces: int, dt: float) -> None:
        self.params = params
        self.max_step = params.max_rate * dt  # largest change allowed in one control step
        self.last = torch.zeros(n_surfaces)  # what was actually commanded last step
        self.tripped = False

    def reset(self) -> None:
        self.last.zero_()
        self.tripped = False

    def apply(self, cmd: Tensor) -> Tensor:
        """Policy command in [-1, 1] per surface -> what is safe to send now. Also in [-1, 1]."""
        if self.tripped:
            return torch.zeros_like(self.last)
        cmd = torch.nan_to_num(cmd, nan=0.0).clamp(-1.0, 1.0)  # NaN -> neutral; never past the limits
        cmd = self.last + (cmd - self.last).clamp(-self.max_step, self.max_step)  # protects servos and model
        self.last = cmd
        return cmd

    def check(self, cm: float) -> None:
        """Trip on an unsafe measured moment. Once tripped, `apply` holds every surface at neutral."""
        if abs(cm) > self.params.cm_trip:
            self.tripped = True
            self.last.zero_()
