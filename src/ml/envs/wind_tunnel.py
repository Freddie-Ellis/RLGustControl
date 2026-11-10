"""The wind tunnel as an RL environment.

The tunnel never stops, so a step is defined by the clock, not the physics. `step(action)`:

    1. sends the (safety-checked) command to the surfaces now
    2. blocks until the next clock tick, while the wing responds to the command and the turbulence
    3. reads every DAQ sample that arrived meanwhile and averages it down to one value per tick
    4. turns the balance moment into Cm and the reward

So each call lasts exactly `spec.dt` of real time and one transition = "what the wing did during one control
period with this command applied". The actor's loop is identical in simulation and in the tunnel. Only this env
knows about time.

    obs        = [~50 taps (normalised), surface commands now in effect]   what the deployed actor has
    privileged = [Cx, Cz, Cm, turbulence level]                             critic only (balance + rig setting)
"""

import time
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from src.hw.clock import FixedRateClock
from src.hw.io import TunnelIO
from src.hw.mock import MockTunnelIO, MockTunnelParams
from src.hw.safety import SafetyLayer, SafetyParams
from src.ml.envs.base import Env, EnvSpec, Trace
from src.ml.envs.rewards import pitch_reward


class WindTunnelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    io: Literal["mock"] = "mock"  # the real DAQ/servo backend gets added here in Phase 4
    max_deflection_deg: float = 15.0  # action +-1 maps to +-this on every surface
    turbulence_min: float = 0.5  # turbulence rms [deg], drawn uniformly per episode
    turbulence_max: float = 3.0
    settle_s: float = 0.5  # wait after setting the turbulence before the episode starts
    tap_scale_pa: float = 500.0  # taps are divided by this so observations are O(1)
    # Calibration: Cm = My / (q S c). In the tunnel these come from the calibration file.
    q_inf: float = 500.0
    area: float = 0.12
    mac: float = 0.35
    # Reward
    cm_scale: float = 0.03  # typical uncontrolled |Cm|; see rewards.pitch_reward
    action_cost: float = 0.05
    rate_cost: float = 0.05
    safety: SafetyParams = SafetyParams()
    mock: MockTunnelParams = MockTunnelParams()


class WindTunnelEnv(Env):
    priv_dim = 4
    priv_names = ("cx", "cz", "cm", "turbulence")

    def __init__(self, spec: EnvSpec, params: WindTunnelParams, n_envs: int = 1, seed: int | None = None) -> None:
        if n_envs != 1:
            raise ValueError("There is one wind tunnel: WindTunnelEnv needs n_envs = 1")
        super().__init__(spec, n_envs, seed)
        self.params = params
        n_taps, n_surfaces = params.mock.n_taps, params.mock.n_surfaces
        self.obs_dim = n_taps + n_surfaces
        self.act_dim = n_surfaces

        # Hardware is opened on the first reset, not here: building the env just to read its dimensions (e.g. in
        # make_policy) must never start the rig.
        self.io: TunnelIO | None = None
        self.clock = FixedRateClock(spec.dt)
        self.safety = SafetyLayer(params.safety, n_surfaces, spec.dt)

        self.level = 0.0
        self.taps = torch.zeros(n_taps)
        self.coeffs = torch.zeros(3)  # Cx, Cz, Cm of the last tick
        self.u = torch.zeros(n_surfaces)  # command in effect (after safety), normalised

    def _open(self) -> TunnelIO:
        match self.params.io:
            case "mock":
                return MockTunnelIO(self.params.mock)

    def _read(self) -> None:
        """Average the samples since the last tick into one reading. Keeps the old one if none arrived."""
        if self.io is None:
            return
        block = self.io.read_block()
        if block.t.numel() == 0:
            return
        p = self.params
        self.taps = block.taps.mean(0)
        fx, fz, my = block.loads.mean(0)
        qs = p.q_inf * p.area
        self.coeffs = torch.stack([fx / qs, fz / qs, my / (qs * p.mac)])

    @property
    def cm(self) -> Tensor:
        return self.coeffs[2]

    def _obs(self) -> Tensor:
        return torch.cat([self.taps / self.params.tap_scale_pa, self.u]).unsqueeze(0)

    def reset(self) -> Tensor:
        p = self.params
        if self.io is None:
            self.io = self._open()
        self.t = 0
        self.safety.reset()
        self.u = torch.zeros(self.act_dim)
        self.io.safe_state()
        # New turbulence level each episode, so the policy sees the whole range it must handle.
        self.level = p.turbulence_min + (p.turbulence_max - p.turbulence_min) * float(torch.rand(1, generator=self.rng))
        self.io.set_turbulence(self.level) # TODO: Communicates with active grid?
        time.sleep(p.settle_s)  # let the flow settle; in the tunnel this is where weights get loaded
        self.io.read_block()  # discard everything from before the episode
        self.clock.start()
        self.clock.wait()
        self._read()
        return self._obs()

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        assert self.io is not None, "call reset() first"
        p = self.params
        prev_u = self.u
        self.u = self.safety.apply(action[0])  # 1. safe command, now
        self.io.write_surfaces(self.u * p.max_deflection_deg)
        self.clock.wait()  # 2. one control period passes in real time
        self._read()  # 3. what happened meanwhile
        self.safety.check(float(self.cm))  # 4. too much moment -> neutral until reset
        if self.safety.tripped:
            self.io.safe_state()
        self.t += 1
        reward = pitch_reward(
            self.cm.unsqueeze(0), self.u.unsqueeze(0), prev_u.unsqueeze(0), p.cm_scale, p.action_cost, p.rate_cost
        )
        return self._obs(), reward

    def privileged(self) -> Tensor:
        level = self.level / self.params.turbulence_max
        return torch.cat([self.coeffs, torch.tensor([level])]).unsqueeze(0)

    def episode_info(self) -> dict[str, float]:
        return {"turbulence_level": self.level}

    def _extras(self) -> dict[str, Tensor]:
        return {
            "cm": self.cm.unsqueeze(0),
            "turbulence": torch.tensor([self.level]),
            "late_ms": torch.tensor([1000.0 * self.clock.last_lateness]),
            "tripped": torch.tensor([float(self.safety.tripped)]),
        }

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        cm = trace.extras["cm"]
        return {
            "cm_rms": float(cm.pow(2).mean().sqrt()),
            "cm_peak": float(cm.abs().max()),
            "late_ms_max": float(trace.extras["late_ms"].max()),
            "missed_ticks": float(self.clock.misses),
        }

    def close(self) -> None:
        if self.io is not None:
            self.io.close()
            self.io = None
        self.clock.close()
