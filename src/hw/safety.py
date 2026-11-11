"""Sits between the policy and the servos. Assumes nothing about the policy: a half-trained network can output
anything, including NaN, full deflection on all surfaces, or a sign flip every step.

The real E-stop is the tunnel's hardware stop. This layer is the software line of defence in front of it.
"""

import math
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

type TripReason = Literal["cm_limit", "sensor_fault"]


class SafetyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rate: float = 20.0  # max change of a normalised command per second (full travel -1 -> 1 in 0.1 s)
    cm_trip: float = 0.3  # |Cm| above this trips: surfaces to neutral until the next reset


class SafetyLayer:
    def __init__(self, params: SafetyParams, n_surfaces: int, dt: float, neutral: Tensor | float = 0.0) -> None:
        self.params = params
        self.n_surfaces = n_surfaces
        # The normalised command that is 0 deg on each surface (RigParams.neutral_action), shape (n_surfaces,).
        self.neutral = torch.as_tensor(neutral, dtype=torch.float32).expand(n_surfaces).clone()
        self.max_step = params.max_rate * dt  # largest change allowed in one control step
        self.last = self._neutral()  # what was actually commanded last step
        self.tripped = False
        self.reason: TripReason | None = None  # why it tripped, None while it has not

    def _neutral(self) -> Tensor:
        return self.neutral.clone()

    def reset(self) -> None:
        self.last = self._neutral()
        self.tripped = False
        self.reason = None

    def apply(self, cmd: Tensor) -> Tensor:
        """Policy command in [-1, 1] per surface -> what is safe to send now. Also in [-1, 1]."""
        if self.tripped:
            return self._neutral()
        cmd = torch.where(cmd.isnan(), self.neutral, cmd).clamp(-1.0, 1.0)  # NaN -> neutral; never past the limits
        cmd = self.last + (cmd - self.last).clamp(-self.max_step, self.max_step)  # protects servos and model
        self.last = cmd
        return cmd.clone()  # the caller's copy: `last` is this layer's own state

    def check(self, cm: float, readings_finite: bool = True) -> None:
        """Trip on an unsafe measured moment, or on readings that are not numbers (a NaN compares False with every
        limit, so it has to be asked for). Once tripped, `apply` holds every surface at neutral."""
        if not readings_finite or not math.isfinite(cm):
            self._trip("sensor_fault")
        elif abs(cm) > self.params.cm_trip:
            self._trip("cm_limit")

    def _trip(self, reason: TripReason) -> None:
        if not self.tripped:  # keep the first reason
            self.tripped = True
            self.reason = reason
            self.last = self._neutral()
