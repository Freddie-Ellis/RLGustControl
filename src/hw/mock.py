"""A fake wind tunnel rig that runs in real wall-clock time, behind the same `TunnelIO` interface as the real one.

It is NOT a delta wing model. It has the features that make the real problem hard, so the env, clock, safety layer
and actor can be tested before going near the tunnel:

    turbulence      random, time-correlated gust angle w(t) whose rms the turbulence device sets
    actuators       12 surfaces with a command delay and a first-order servo lag
    aerodynamics    a first-order lag between (gust, deflections) and the loads/pressures
    taps            ~50 pressures: linear in the same lagged state, each surface felt most by nearby taps, + noise
    balance         Fx, Fz, My from the same state, + noise

Real DAQs stream samples continuously in hardware. The mock does the equivalent lazily: whenever it is called, it
first simulates every sample due between the last call and now, so its output is what a real stream would hold.
Retune its constants from open-loop tunnel data (Phase 4) so sim pretraining matches the rig.
"""

import math
import time
from collections import deque

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from src.hw.io import SensorBlock


class MockTunnelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daq_rate_hz: float = 1000.0
    n_taps: int = 50
    n_surfaces: int = 12
    actuator_delay_s: float = 0.004  # command -> servo starts moving
    servo_tau_s: float = 0.02  # servo first-order lag
    aero_tau_s: float = 0.03  # flow/loads lag behind gust and deflection
    turb_tau_s: float = 0.1  # correlation time of the turbulence
    q_inf: float = 500.0  # dynamic pressure [Pa], ~29 m/s
    area: float = 0.12  # reference area [m^2]
    mac: float = 0.35  # mean aerodynamic chord [m]
    alpha0_deg: float = 8.0
    cm_per_gust_deg: float = 0.01  # how strongly turbulence disturbs Cm
    tap_noise_pa: float = 2.0
    load_noise_n: float = 0.2
    seed: int = 0


class MockTunnelIO:
    def __init__(self, params: MockTunnelParams) -> None:
        p = params
        self.params = p
        self.n_taps = p.n_taps
        self.n_surfaces = p.n_surfaces
        self.dt = 1.0 / p.daq_rate_hz
        self.rng = torch.Generator().manual_seed(p.seed)

        # Fixed "geometry", seeded so every run sees the same wing.
        tap_pos = torch.rand(p.n_taps, generator=self.rng)  # spanwise position of each tap, 0..1
        surf_pos = torch.linspace(0.0, 1.0, p.n_surfaces)  # spanwise position of each surface
        self.cm_per_deg = -0.0025 * (0.6 + 0.8 * surf_pos)  # outboard surfaces have a longer moment arm
        self.cz_per_deg = torch.full((p.n_surfaces,), 0.004)
        near = torch.exp(-(((tap_pos[:, None] - surf_pos[None, :]) / 0.15) ** 2))  # [n_taps, n_surfaces]
        self.cp0 = -1.0 + 1.5 * torch.rand(p.n_taps, generator=self.rng)
        self.cp_per_gust = 0.03 * (0.5 + torch.rand(p.n_taps, generator=self.rng))
        self.cp_per_deg = 0.015 * near  # a surface is felt most by the taps near it

        # State
        self.t = 0.0  # simulated time [s], kept equal to wall-clock time since open
        self.t0 = time.perf_counter()
        self.level = 0.0  # turbulence rms [deg]
        self.w = 0.0  # gust angle [deg]
        self.cmd = torch.zeros(p.n_surfaces)  # command the servos are tracking [deg]
        self.defl = torch.zeros(p.n_surfaces)  # actual deflection [deg]
        self.aero = torch.zeros(1 + p.n_surfaces)  # lagged [gust, deflections] the flow responds to
        self.pending: deque[tuple[float, Tensor]] = deque()  # (time it takes effect, command)
        self._t_buf: list[float] = []
        self._aero_buf: list[Tensor] = []

    def _advance_to_now(self) -> None:
        """Simulate every DAQ sample due up to the current wall-clock time."""
        p = self.params
        now = time.perf_counter() - self.t0
        k_turb = self.dt / p.turb_tau_s
        k_servo = self.dt / p.servo_tau_s
        k_aero = self.dt / p.aero_tau_s
        while self.t + self.dt <= now:
            self.t += self.dt
            while self.pending and self.pending[0][0] <= self.t:
                self.cmd = self.pending.popleft()[1]
            # Ornstein-Uhlenbeck turbulence: correlated over turb_tau_s, rms = level.
            noise = float(torch.randn(1, generator=self.rng))
            self.w += -self.w * k_turb + self.level * math.sqrt(2.0 * k_turb) * noise
            self.defl = self.defl + (self.cmd - self.defl) * k_servo
            target = torch.cat([torch.tensor([self.w]), self.defl])
            self.aero = self.aero + (target - self.aero) * k_aero
            self._t_buf.append(self.t)
            self._aero_buf.append(self.aero)

    def read_block(self) -> SensorBlock:
        self._advance_to_now()
        p = self.params
        m = len(self._t_buf)
        t = torch.tensor(self._t_buf)
        aero = torch.stack(self._aero_buf) if m else torch.zeros(0, 1 + p.n_surfaces)
        self._t_buf, self._aero_buf = [], []
        gust, defl = aero[:, 0], aero[:, 1:]

        cp = self.cp0 + gust[:, None] * self.cp_per_gust + defl @ self.cp_per_deg.T
        taps = p.q_inf * cp + p.tap_noise_pa * torch.randn(m, p.n_taps, generator=self.rng)

        alpha = p.alpha0_deg + gust
        cx = 0.02 + 0.0005 * alpha**2
        cz = 0.05 * alpha + defl @ self.cz_per_deg
        cm = 0.01 + p.cm_per_gust_deg * gust + defl @ self.cm_per_deg
        qs = p.q_inf * p.area
        loads = torch.stack([qs * cx, qs * cz, qs * p.mac * cm], dim=-1)
        loads = loads + p.load_noise_n * torch.randn(m, 3, generator=self.rng)
        return SensorBlock(t=t, taps=taps, loads=loads)

    def write_surfaces(self, deg: Tensor) -> None:
        self._advance_to_now()  # samples before now used the old command
        self.pending.append((self.t + self.params.actuator_delay_s, deg.detach().clone()))

    def set_turbulence(self, level: float) -> None:
        self._advance_to_now()
        self.level = level

    def safe_state(self) -> None:
        self._advance_to_now()
        self.pending.clear()
        self.pending.append((self.t + self.params.actuator_delay_s, torch.zeros(self.n_surfaces)))

    def close(self) -> None:
        self.safe_state()
