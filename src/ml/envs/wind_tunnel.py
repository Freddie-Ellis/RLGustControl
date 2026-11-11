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
    action     = one value in [-1, 1] per surface, mapped onto its travel (CONFIG.rig: 0..20 deg, so -1 is neutral)
    privileged = [Cx, Cz, Cm (each scaled to O(1))]                         critic only (the balance)

The turbulence is whatever the tunnel gives: the env neither sets it nor is told its level.

A safety trip ends the episode (`terminated`): the surfaces go to neutral and stay there until the next reset.
"""

import time
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from src.config import CONFIG, RigParams
from src.hw.clock import FixedRateClock
from src.hw.io import TunnelIO
from src.hw.mock import MockTunnelIO, MockTunnelParams
from src.hw.safety import SafetyLayer, SafetyParams
from src.ml.envs.base import Env, EnvSpec, Trace
from src.ml.envs.rewards import pitch_reward

_OLD_TURBULENCE_KEYS = ("turbulence_min", "turbulence_max", "eval_turbulence")


class WindTunnelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    io: Literal["mock"] = "mock"  # the real DAQ/servo backend gets added here in Phase 4
    # Taps, surfaces, their travel and the calibration. Defined once, in src/config.py; this is the run's copy.
    rig: RigParams = Field(default_factory=lambda: CONFIG.rig.model_copy())
    settle_s: float = 0.5  # wait with the surfaces at neutral before the episode starts
    tap_scale_pa: float = 500.0  # taps are divided by this so observations are O(1)
    # Typical sizes of Cx and Cz, which the critic's copies are divided by so its inputs are O(1) like the taps.
    cx_scale: float = 0.05
    cz_scale: float = 0.4
    # Reward
    cm_scale: float = 0.03  # typical uncontrolled |Cm|; see rewards.pitch_reward. Also scales the critic's Cm
    action_cost: float = 0.05
    rate_cost: float = 0.05
    # Taken off the reward of the step that trips the safety layer. Every step's reward is negative, so without
    # it ending the episode early would look better than carrying on. About the cost of the ~1 / (1 - gamma)
    # steps of uncontrolled flow the critic no longer counts.
    trip_penalty: float = 100.0
    safety: SafetyParams = SafetyParams()
    mock: MockTunnelParams = MockTunnelParams()

    @model_validator(mode="before")
    @classmethod
    def _rig_from_old_layout(cls, data: Any) -> Any:
        """Configs saved before `rig` existed kept q_inf, area and mac here, the tap and surface counts under
        `mock`, and a symmetric travel as `max_deflection_deg`. Move them, so old sessions and runs still load as
        what they were. They also set a turbulence level per episode, which the env no longer does: dropped."""
        if not isinstance(data, dict):
            return data
        if any(k in data for k in _OLD_TURBULENCE_KEYS):
            data = {k: v for k, v in data.items() if k not in _OLD_TURBULENCE_KEYS}
        mock = data.get("mock")
        mock = dict(mock) if isinstance(mock, dict) else {}
        old_keys = (*RigParams.model_fields, "max_deflection_deg")
        if not any(k in data or k in mock for k in old_keys):
            return data
        data = dict(data)
        rig = dict(data.get("rig") or {})
        if (travel := data.pop("max_deflection_deg", None)) is not None:
            rig.setdefault("actuator_min_deg", -travel)
            rig.setdefault("actuator_max_deg", travel)
        for key in RigParams.model_fields:
            for old in (data, mock):  # this level first: it is the calibration Cm was computed with
                if key in old:
                    rig.setdefault(key, old.pop(key))
        data["rig"] = rig
        if "mock" in data:
            data["mock"] = mock
        return data


class WindTunnelEnv(Env):
    priv_dim = 3
    priv_names = ("cx", "cz", "cm")

    def __init__(self, spec: EnvSpec, params: WindTunnelParams, n_envs: int = 1, seed: int | None = None) -> None:
        if n_envs != 1:
            raise ValueError("There is one wind tunnel: WindTunnelEnv needs n_envs = 1")
        super().__init__(spec, n_envs, seed)
        self.params = params
        n_taps, n_surfaces = params.rig.n_taps, params.rig.n_surfaces
        self.obs_dim = n_taps + n_surfaces
        self.act_dim = n_surfaces

        # Hardware is opened on the first reset, not here: building the env just to read its dimensions (e.g. in
        # make_policy) must never start the rig.
        self.io: TunnelIO | None = None
        self.clock = FixedRateClock(spec.dt)
        self.safety = SafetyLayer(params.safety, n_surfaces, spec.dt, neutral=params.rig.neutral_action)

        self.taps = torch.zeros(n_taps)
        self.coeffs = torch.zeros(3)  # Cx, Cz, Cm of the last tick
        self.u = self._neutral()  # command in effect (after safety), normalised
        self.samples = 0  # DAQ samples averaged into the last tick

        # Totals over the current episode, for episode_info.
        self._steps = 0
        self._limited = 0  # surface commands (one per surface per step) the safety layer changed
        self._gap = 0.0  # summed |applied - requested| over those commands, in action units
        self._late_max = 0.0  # worst tick lateness [s]
        self._samples_min: int | None = None

    def _open(self) -> TunnelIO:
        match self.params.io:
            case "mock":
                return MockTunnelIO(self.params.rig, self.params.mock)

    def _read(self) -> bool:
        """Average the samples since the last tick into one reading. Keeps the old one if none arrived.

        False if the new reading is not finite (a sensor or DAQ fault). The old one is kept then too, so nothing
        non-finite reaches the policy or the replay buffer; the caller trips the safety layer."""
        if self.io is None:
            return True
        block = self.io.read_block()
        self.samples = block.t.numel()
        if self.samples == 0:
            return True
        taps, loads = block.taps.mean(0), block.loads.mean(0)
        if not (torch.isfinite(taps).all() and torch.isfinite(loads).all()):
            return False
        rig = self.params.rig
        self.taps = taps
        fx, fz, my = loads
        qs = rig.q_inf * rig.area
        self.coeffs = torch.stack([fx / qs, fz / qs, my / (qs * rig.mac)])
        return True

    @property
    def cm(self) -> Tensor:
        return self.coeffs[2]

    def _neutral(self) -> Tensor:
        """The normalised command that holds every surface at 0 deg."""
        return self.params.rig.neutral_action

    def zero_action(self, obs: Tensor) -> Tensor:
        """Uncontrolled baseline: every surface at neutral, which is not action 0 when the travel is one-sided."""
        return self._neutral().expand(obs.shape[0], -1).clone()

    def _obs(self) -> Tensor:
        return torch.cat([self.taps / self.params.tap_scale_pa, self.u]).unsqueeze(0)

    def reset(self) -> Tensor:
        if self.io is None:
            self.io = self._open()
        self.t = 0
        self._steps, self._limited, self._gap, self._late_max, self._samples_min = 0, 0, 0.0, 0.0, None
        self.safety.reset()
        self.u = self._neutral()
        self.io.safe_state()
        time.sleep(self.params.settle_s)  # let the flow settle; in the tunnel this is where weights get loaded
        self.io.read_block()  # discard everything from before the episode
        self.clock.start()
        self.clock.wait()
        self._read()  # a fault here shows up again on the first step, which trips
        return self._obs()

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        assert self.io is not None, "call reset() first"
        p = self.params
        prev_u = self.u
        was_tripped = self.safety.tripped
        applied = self.safety.apply(action[0])  # 1. safe command, now
        self.u = applied
        self.io.write_surfaces(p.rig.to_deg(applied))
        self.clock.wait()  # 2. one control period passes in real time
        readings_finite = self._read()  # 3. what happened meanwhile
        self.safety.check(float(self.cm), readings_finite)  # 4. too much moment or a dead sensor -> neutral
        self.t += 1
        self._steps += 1
        gap = torch.nan_to_num((applied - action[0]).abs(), nan=2.0)  # a NaN request counts as the full range
        self._limited += int((gap > 1e-6).sum())
        self._gap += float(gap.sum())
        self._late_max = max(self._late_max, self.clock.last_lateness)
        self._samples_min = self.samples if self._samples_min is None else min(self._samples_min, self.samples)

        # The reward is for the command that was applied during this step, whatever happens to it next. Effort and
        # rate are counted in deflection from neutral, so holding the surfaces flush costs nothing.
        deflection, prev_deflection = p.rig.deflection_fraction(applied), p.rig.deflection_fraction(prev_u)
        reward = pitch_reward(
            self.cm.unsqueeze(0), deflection.unsqueeze(0), prev_deflection.unsqueeze(0),
            p.cm_scale, p.action_cost, p.rate_cost,
        )
        if self.safety.tripped:
            self.io.safe_state()
            self.u = self._neutral()  # the command now in effect, which the next observation reports
            if not was_tripped:
                reward = reward - p.trip_penalty
        return self._obs(), reward

    def privileged(self) -> Tensor:
        p = self.params
        cx, cz, cm = self.coeffs
        return torch.stack([cx / p.cx_scale, cz / p.cz_scale, cm / p.cm_scale]).unsqueeze(0)

    def terminated(self) -> Tensor:
        return torch.tensor([float(self.safety.tripped)])

    def episode_info(self) -> dict[str, float]:
        return {
            "tripped": float(self.safety.tripped),
            "sensor_fault": float(self.safety.reason == "sensor_fault"),
            "missed_ticks": float(self.clock.misses),
            "late_ms_max": 1000.0 * self._late_max,
            "samples_min": float(self._samples_min or 0),  # fewest DAQ samples in one tick; 0 = the DAQ stalled
            # How much the safety layer (rate limit, clamp) stood between the policy and the surfaces: the share
            # of surface commands it changed, and by how much on average over all of them (action units, range 2).
            "limited_frac": self._limited / max(self._steps * self.act_dim, 1),
            "limit_gap": self._gap / max(self._steps * self.act_dim, 1),
        }

    def extras(self) -> dict[str, Tensor]:
        return {
            "cm": self.cm.unsqueeze(0),  # raw Cm, as measured: the privileged copy is scaled for the critic
            "late_ms": torch.tensor([1000.0 * self.clock.last_lateness]),
            "samples": torch.tensor([float(self.samples)]),
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
