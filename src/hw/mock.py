"""A fake wind tunnel rig that runs in real wall-clock time, behind the same `TunnelIO` interface as the real one.

It is NOT a delta wing model. It has the features that make the real problem hard, so the env, clock, safety layer
and actor can be tested before going near the tunnel:

    turbulence      random, time-correlated gust angle w(t) of a fixed rms: the tunnel's own, which nobody sets
    actuators       12 surfaces with a command delay and a first-order servo lag, and the watchdog `TunnelIO` asks
                    of every backend (neutral when commands stop coming)
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

from src.config import RigParams
from src.hw.io import SensorBlock


class MockTunnelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daq_rate_hz: float = 1000.0
    actuator_delay_s: float = 0.004  # command -> servo starts moving
    servo_tau_s: float = 0.02  # servo first-order lag
    aero_tau_s: float = 0.03  # flow/loads lag behind gust and deflection
    turb_tau_s: float = 0.1  # correlation time of the turbulence
    turbulence_deg: float = 2.0  # rms gust angle [deg] of the flow
    watchdog_s: float = 0.1  # no command for this long -> surfaces to neutral (see TunnelIO: the watchdog)
    alpha0_deg: float = 8.0
    cm_per_gust_deg: float = 0.01  # how strongly turbulence disturbs Cm
    tap_noise_pa: float = 2.0
    load_noise_n: float = 0.2
    seed: int = 0


class MockTunnelIO:
    def __init__(self, rig: RigParams, params: MockTunnelParams) -> None:
        p = params
        self.rig = rig
        self.params = p
        self.n_taps = rig.n_taps
        self.n_surfaces = rig.n_surfaces
        self.dt = 1.0 / p.daq_rate_hz
        self.rng = torch.Generator().manual_seed(p.seed)

        # Fixed "geometry", seeded so every run sees the same wing.
        tap_pos = torch.rand(rig.n_taps, generator=self.rng)  # spanwise position of each tap, 0..1
        surf_pos = torch.linspace(0.0, 1.0, rig.n_surfaces)  # spanwise position of each surface
        # Outboard surfaces have a longer moment arm. Neighbouring surfaces pitch the wing opposite ways: the travel
        # is one-sided (0..max), so surfaces that all pushed Cm the same way could only cancel half of the gusts.
        sign = torch.where(torch.arange(rig.n_surfaces) % 2 == 0, -1.0, 1.0)
        self.cm_per_deg = sign * 0.0025 * (0.6 + 0.8 * surf_pos)
        self.cz_per_deg = torch.full((rig.n_surfaces,), 0.004)
        near = torch.exp(-(((tap_pos[:, None] - surf_pos[None, :]) / 0.15) ** 2))  # [n_taps, n_surfaces]
        self.cp0 = -1.0 + 1.5 * torch.rand(rig.n_taps, generator=self.rng)
        self.cp_per_gust = 0.03 * (0.5 + torch.rand(rig.n_taps, generator=self.rng))
        self.cp_per_deg = 0.015 * near  # a surface is felt most by the taps near it

        # State
        self.t = 0.0  # simulated time [s], kept equal to wall-clock time since open
        self.t0 = time.perf_counter()
        self.level = p.turbulence_deg  # turbulence rms [deg]
        self.w = 0.0  # gust angle [deg]
        self._neutral = torch.zeros(rig.n_surfaces)
        self.cmd = self._neutral  # command the servos are tracking [deg]
        self.defl = torch.zeros(rig.n_surfaces)  # actual deflection [deg]
        self.aero = torch.zeros(1 + rig.n_surfaces)  # lagged [gust, deflections] the flow responds to
        self.pending: deque[tuple[float, Tensor]] = deque()  # (time it takes effect, command)
        self.last_command_t = 0.0  # when the last command arrived, for the watchdog
        self.watchdog_trips = 0  # times the watchdog had to take the surfaces to neutral
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
            if self.t - self.last_command_t > p.watchdog_s and self.cmd is not self._neutral:
                # Nobody is commanding the surfaces any more (the control loop hung or stopped): neutral.
                self.pending.clear()
                self.cmd = self._neutral
                self.watchdog_trips += 1
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
        p, rig = self.params, self.rig
        m = len(self._t_buf)
        t = torch.tensor(self._t_buf)
        aero = torch.stack(self._aero_buf) if m else torch.zeros(0, 1 + rig.n_surfaces)
        self._t_buf, self._aero_buf = [], []
        gust, defl = aero[:, 0], aero[:, 1:]

        cp = self.cp0 + gust[:, None] * self.cp_per_gust + defl @ self.cp_per_deg.T
        taps = rig.q_inf * cp + p.tap_noise_pa * torch.randn(m, rig.n_taps, generator=self.rng)

        alpha = p.alpha0_deg + gust
        cx = 0.02 + 0.0005 * alpha**2
        cz = 0.05 * alpha + defl @ self.cz_per_deg
        cm = 0.01 + p.cm_per_gust_deg * gust + defl @ self.cm_per_deg
        qs = rig.q_inf * rig.area
        loads = torch.stack([qs * cx, qs * cz, qs * rig.mac * cm], dim=-1)
        loads = loads + p.load_noise_n * torch.randn(m, 3, generator=self.rng)
        return SensorBlock(t=t, taps=taps, loads=loads)

    def write_surfaces(self, deg: Tensor) -> None:
        self._advance_to_now()  # samples before now used the old command
        self.last_command_t = self.t
        self.pending.append((self.t + self.params.actuator_delay_s, deg.detach().clone()))

    def safe_state(self) -> None:
        self._advance_to_now()
        self.last_command_t = self.t
        self.pending.clear()
        self.pending.append((self.t + self.params.actuator_delay_s, self._neutral))

    def close(self) -> None:
        self.safe_state()
