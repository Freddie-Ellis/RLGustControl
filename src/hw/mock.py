"""A fake wind tunnel rig that runs in real wall-clock time, behind the same `TunnelIO` interface as the real one.

It is NOT a delta wing model. It has the features that make the real problem hard, so the env, clock, safety layer
and actor can be tested before going near the tunnel:

    turbulence      random, time-correlated gust angle w(t) of a fixed rms: the tunnel's own, which nobody sets; and an
                    independent antisymmetric one that rolls the wing
    actuators       12 surfaces with a command delay and a first-order servo lag, and the watchdog `TunnelIO` asks
                    of every backend (neutral when commands stop coming)
    aerodynamics    a first-order lag between (gust, deflections) and the loads/pressures
    taps            ~50 pressures: linear in the same lagged state, each surface felt most by nearby taps, + noise
    load cell       all six forces and moments from the same state, + noise: lift, drag and pitch respond to the gust
                    and every surface, roll to the antisymmetric gust and to surfaces by which side they are on

Real DAQs stream samples continuously in hardware. The mock does the equivalent lazily: whenever it is called, it
first simulates every sample due between the last call and now, so its output is what a real stream would hold.
TODO: retune its constants from open-loop tunnel data (Phase 4) so sim pretraining matches the rig.
"""

import math
import time
from collections import deque
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, model_validator
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
    roll_turbulence_deg: float = 1.0  # rms of the antisymmetric gust [deg] (one wing up, the other down)
    watchdog_s: float = 0.1  # no command for this long -> surfaces to neutral (see TunnelIO: the watchdog)
    # The simulated tunnel's operating point, which it gives with every read. It can drift slowly (a sine of the
    # given amplitude and period), as the real tunnel's speed and the model's pitch do while running.
    q_pa: float = 500.0  # dynamic pressure [Pa]
    alpha_deg: float = 8.0  # angle of attack [deg]
    q_drift_pa: float = 0.0
    alpha_drift_deg: float = 0.0
    drift_period_s: float = 60.0
    cm_per_gust_deg: float = 0.01  # how strongly turbulence disturbs Cm
    roll_per_gust_deg: float = 0.004  # how strongly the antisymmetric gust disturbs C_l
    tap_noise_pa: float = 2.0
    load_noise_n: float = 0.2  # on every force [N] and moment [N m]
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data: Any) -> Any:
        """Old configs called the angle of attack alpha0_deg."""
        if isinstance(data, dict) and "alpha0_deg" in data:
            data = {("alpha_deg" if k == "alpha0_deg" else k): v for k, v in data.items()}
        return data


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
        # Roll: the first half of the span is one wing, the second the other; outboard surfaces roll it most. Not tied
        # to the alternating pitch signs, so pitch and roll can be set independently.
        self.roll_per_deg = 0.004 * (2.0 * surf_pos - 1.0)
        near = torch.exp(-(((tap_pos[:, None] - surf_pos[None, :]) / 0.15) ** 2))  # [n_taps, n_surfaces]
        self.cp0 = -1.0 + 1.5 * torch.rand(rig.n_taps, generator=self.rng)
        self.cp_per_gust = 0.03 * (0.5 + torch.rand(rig.n_taps, generator=self.rng))
        self.cp_per_deg = 0.015 * near  # a surface is felt most by the taps near it

        # State
        self.t = 0.0  # simulated time [s], kept equal to wall-clock time since open
        self.t0 = time.perf_counter()
        self.level = p.turbulence_deg  # turbulence rms [deg]
        self.w = 0.0  # gust angle [deg]
        self.w_roll = 0.0  # antisymmetric gust [deg]
        self._neutral = torch.zeros(rig.n_surfaces)
        self.cmd = self._neutral  # command the servos are tracking [deg]
        self.defl = torch.zeros(rig.n_surfaces)  # actual deflection [deg]
        self.aero = torch.zeros(2 + rig.n_surfaces)  # lagged [gust, roll gust, deflections] the flow responds to
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
            noise, noise_roll = torch.randn(2, generator=self.rng).tolist()
            self.w += -self.w * k_turb + self.level * math.sqrt(2.0 * k_turb) * noise
            self.w_roll += -self.w_roll * k_turb + p.roll_turbulence_deg * math.sqrt(2.0 * k_turb) * noise_roll
            self.defl = self.defl + (self.cmd - self.defl) * k_servo
            target = torch.cat([torch.tensor([self.w, self.w_roll]), self.defl])
            self.aero = self.aero + (target - self.aero) * k_aero
            self._t_buf.append(self.t)
            self._aero_buf.append(self.aero)

    def read_block(self) -> SensorBlock:
        self._advance_to_now()
        p, rig = self.params, self.rig
        m = len(self._t_buf)
        t = torch.tensor(self._t_buf)
        aero = torch.stack(self._aero_buf) if m else torch.zeros(0, 2 + rig.n_surfaces)
        self._t_buf, self._aero_buf = [], []
        gust, roll_gust, defl = aero[:, 0], aero[:, 1], aero[:, 2:]

        # The operating point at each sample: slow drift of the tunnel's speed and the rig's pitch.
        phase = torch.sin(2.0 * math.pi * t / p.drift_period_s)
        q = p.q_pa + p.q_drift_pa * phase
        attitude = p.alpha_deg + p.alpha_drift_deg * phase

        cp = self.cp0 + gust[:, None] * self.cp_per_gust + defl @ self.cp_per_deg.T
        cp = cp + 0.01 * roll_gust[:, None] * (self.cp_per_gust - self.cp_per_gust.mean())  # the taps feel it too
        cp = cp + 0.05 * (attitude - p.alpha_deg)[:, None] * self.cp_per_gust  # and the pitch, like a slow gust
        taps = q[:, None] * cp + p.tap_noise_pa * torch.randn(m, rig.n_taps, generator=self.rng)

        # Coefficients as the CFD defines them (wind axes for forces; see src/ml/envs/wind_tunnel.py), then into the
        # model frame's forces and moments, which is what a load cell reports.
        alpha = attitude + gust
        c_drag = 0.02 + 0.0005 * alpha**2
        c_lift = 0.05 * alpha + defl @ self.cz_per_deg
        c_pitch = 0.01 + p.cm_per_gust_deg * gust + defl @ self.cm_per_deg
        c_roll = p.roll_per_gust_deg * roll_gust + defl @ self.roll_per_deg
        c_yaw = 0.1 * c_roll
        c_side = torch.zeros_like(c_roll)
        a = torch.deg2rad(attitude)  # lift and drag are about the freestream, at the rig's attitude
        qs = q * rig.area
        loads = torch.stack([
            qs * (c_drag * a.cos() - c_lift * a.sin()),  # Fx (aft)
            qs * c_side,  # Fy (starboard)
            qs * (c_lift * a.cos() + c_drag * a.sin()),  # Fz (up)
            -qs * rig.span * c_roll,  # Mx: C_l = -Mx / (q S b)
            qs * rig.mac * c_pitch,  # My: C_m = My / (q S c)
            -qs * rig.span * c_yaw,  # Mz: C_n = -Mz / (q S b)
        ], dim=-1)
        loads = loads + p.load_noise_n * torch.randn(m, 6, generator=self.rng)
        # What the tunnel reports now: its operating point at the time of the read.
        now = math.sin(2.0 * math.pi * self.t / p.drift_period_s)
        return SensorBlock(t=t, taps=taps, loads=loads, q=p.q_pa + p.q_drift_pa * now,
                           alpha=p.alpha_deg + p.alpha_drift_deg * now)

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
