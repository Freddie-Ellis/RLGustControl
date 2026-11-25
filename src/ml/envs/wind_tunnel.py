"""The wind tunnel as an RL environment.

The tunnel never stops, so a step is defined by the clock, not the physics. `step(action)`:

    1. sends the (safety-checked) command to the surfaces now
    2. blocks until the next clock tick, while the wing responds to the command and the turbulence
    3. reads every DAQ sample that arrived meanwhile and averages it down to one value per tick
    4. turns the 6-axis load cell into lift, pitch and roll coefficients (`LOADS`), and those into the reward

So each call lasts exactly `spec.dt` of real time and one transition = "what the wing did during one control
period with this command applied". The actor's loop is identical in simulation and in the tunnel. Only this env
knows about time.

    obs        = [targets of the tracked loads (normalised), angle of attack and dynamic pressure (normalised),
                  ~50 taps as Cp, surface commands now in effect]                 what the deployed actor has
    action     = one value in [-1, 1] per surface, mapped onto its travel (CONFIG.rig)
    privileged = [C_L, C_m, C_l (each over LOAD_SCALES)]                  critic only (the load cell)

Tracking: `tracking` names the loads the controller holds at a target, any mix of `LOADS`, each with the scale its
error is measured in, a weight and the range its target is drawn from (once per episode, and again every
`retarget_every_s` if set). See rewards.tracking_reward. A target range of one value holds that load there: the
default holds C_m at 0, the original pitch task.

The targets are part of the observation, so one trained policy holds any target inside the ranges it was trained on.
To choose them: `eval_targets` in the config for evaluation and baseline episodes (the actor pins them), or
`set_targets({"C_L": 0.45})` on the env from code, at any time, until set again.

The turbulence is whatever the tunnel gives: the env neither sets it nor is told its level.

The dynamic pressure q and the angle of attack change as the tunnel runs, so both are measured with every DAQ sample
and averaged per tick like everything else. The loads are made coefficients with that tick's q and alpha, and the
taps are given to the policy as Cp = p / q (as the CFD and the autoencoder have them), so a change of tunnel speed
does not change what it sees. The policy is also given alpha and q themselves: the wing's response depends on them.
A q below MIN_Q_PA (tunnel stopped or the pitot blocked) trips the safety layer, since no coefficient means anything
then.

A safety trip ends the episode (`terminated`): the surfaces go to neutral and stay there until the next reset.
"""

import time
from collections.abc import Mapping
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
from src.ml.envs.rewards import tracking_reward

_OLD_TURBULENCE_KEYS = ("turbulence_min", "turbulence_max", "eval_turbulence")

# What `load_coefficients` makes of the 6-axis load cell: lift, drag and side force (wind axes), then rolling,
# pitching and yawing moment, as the CFD defines them (src/cfd/solve_report.py). Lower-case l is roll.
# TODO: decide whether the controller should use the CFD's definitions at all. The CFD's are for comparing with it
# (and training on it); for control, body axes (normal force rather than lift, which turns with alpha), the load
# cell's own moment centre, or the rig's reference area and chord may suit better. Changing them here changes every
# target range and scale with them.
COEFFICIENTS: tuple[str, ...] = ("C_L", "C_D", "C_Y", "C_l", "C_m", "C_n")
type LoadName = Literal["C_L", "C_m", "C_l"]
# The loads the controller may hold at a target, any mix of them: lift, pitch and roll. The critic gets these three.
LOADS: tuple[LoadName, ...] = ("C_L", "C_m", "C_l")
# Typical size of each load, which the critic's copies are divided by so its inputs are O(1) like the taps.
LOAD_SCALES: dict[LoadName, float] = {"C_L": 0.4, "C_m": 0.03, "C_l": 0.02}  # TODO: from baseline episodes
# The operating point in the observation: alpha and q divided by these, so they are O(1) like the rest.
ALPHA_SCALE_DEG: float = 10.0
Q_SCALE_PA: float = 500.0  # TODO: the tunnel's usual dynamic pressure
MIN_Q_PA: float = 50.0  # TODO: set from the tunnel's lowest running speed. Below it the reading is a fault: trips


class LoadTarget(BaseModel):
    """How one tracked load enters the reward and where its target comes from."""

    model_config = ConfigDict(extra="forbid")

    # Typical uncontrolled excursion of the load in gusts: an error this size costs `weight` (rewards.tracking_reward).
    # TODO: set it from baseline episodes (surfaces at neutral). The excursion grows with alpha (the lift slope, and
    # the vortices near stall), so measure it across the alpha range the session runs and take one value for all,
    # e.g. the median rms; it only sets the reward's units, so the right order of magnitude matters more than
    # precision. If one value fits badly across alpha, make it a function of alpha instead.
    scale: float = Field(gt=0)
    weight: float = Field(default=1.0, ge=0)
    # The target is drawn uniformly from [low, high]; low == high holds the load at that value. Keep it to what the
    # surfaces can reach at the rig's angle of attack, or the policy learns it can never win.
    target: tuple[float, float] = (0.0, 0.0)

    @model_validator(mode="after")
    def _ordered(self) -> LoadTarget:
        if self.target[0] > self.target[1]:
            raise ValueError(f"target range {self.target} must be (low, high)")
        return self


class WindTunnelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    io: Literal["mock"] = "mock"  # the real DAQ/servo backend gets added here in Phase 4
    # Taps, surfaces, their travel and the calibration. Defined once, in src/config.py; this is the run's copy.
    rig: RigParams = Field(default_factory=lambda: CONFIG.rig.model_copy())
    settle_s: float = 0.5  # wait with the surfaces at neutral before the episode starts
    # Reward: the loads held at a target (any of LOADS; see the module docstring), and the action costs.
    tracking: dict[LoadName, LoadTarget] = Field(default_factory=lambda: {"C_m": LoadTarget(scale=0.03)})
    retarget_every_s: float | None = None  # draw new targets this often within an episode (step changes); None: once
    # Targets that evaluation and baseline episodes pin, so every eval in a session is scored on the same task;
    # tracked loads not listed are drawn as in training. Each must be tracked and inside its target range.
    eval_targets: dict[LoadName, float] = Field(default_factory=dict)
    action_cost: float = 0.05
    rate_cost: float = 0.05
    # Taken off the reward of the step that trips the safety layer. Every step's reward is negative, so without
    # it ending the episode early would look better than carrying on. About the cost of the ~1 / (1 - gamma)
    # steps of uncontrolled flow the critic no longer counts.
    trip_penalty: float = 100.0
    safety: SafetyParams = SafetyParams()
    mock: MockTunnelParams = MockTunnelParams()

    @model_validator(mode="after")
    def _something_tracked(self) -> WindTunnelParams:
        if not self.tracking:
            raise ValueError(f"track at least one load: add a [tracking.<load>] table, <load> one of {LOADS}")
        check_targets(self.eval_targets, self.tracking)
        return self

    @model_validator(mode="before")
    @classmethod
    def _from_old_pitch_task(cls, data: Any) -> Any:
        """Configs saved before `tracking` held Cm at 0 with `cm_scale`, and scaled the critic's Cx and Cz with
        `cx_scale` and `cz_scale`: the same task as tracking C_m, and the scales now in LOAD_SCALES. They also scaled
        the taps by a fixed `tap_scale_pa` and took q as a constant `q_inf`, both now measured: dropped."""
        if not isinstance(data, dict):
            return data
        data = {k: v for k, v in data.items() if k not in ("cx_scale", "cz_scale", "tap_scale_pa", "q_inf")}
        if isinstance(data.get("mock"), dict):
            data["mock"] = {k: v for k, v in data["mock"].items() if k != "q_inf"}
        if (cm_scale := data.pop("cm_scale", None)) is not None:
            data.setdefault("tracking", {"C_m": {"scale": cm_scale}})
        return data

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

    @property
    def tracked(self) -> tuple[LoadName, ...]:
        """The tracked loads, in LOADS order: the order of the targets in the observation and of the reward's terms."""
        return tuple(name for name in LOADS if name in self.tracking)


def check_targets(targets: Mapping[str, float], tracking: Mapping[str, LoadTarget]) -> None:
    """Raises unless every target is for a tracked load and inside the range it was trained on: the policy has never
    seen a target outside it, and its observation of one would be outside -1..1."""
    for name, value in targets.items():
        if name not in tracking:
            raise ValueError(f"{name} is not tracked (tracked: {sorted(tracking)}): no target to set")
        low, high = tracking[name].target
        if not low <= value <= high:
            raise ValueError(f"{name} target {value} is outside its trained range [{low}, {high}]")


def load_coefficients(loads: Tensor, q: Tensor, alpha_deg: Tensor, rig: RigParams) -> Tensor:
    """The six load cell readings [..., 6] (Fx, Fy, Fz [N], Mx, My, Mz [N m] in the model frame: x aft, y starboard,
    z up), at dynamic pressure q [...] [Pa] and angle of attack alpha_deg [...] -> the coefficients in COEFFICIENTS
    [..., 6], as src/cfd/solve_report.py defines them. Lift and drag are across and along the freestream at that
    angle of attack; positive roll is right wing down, yaw nose right."""
    fx, fy, fz, mx, my, mz = loads.unbind(-1)
    qs = q * rig.area
    a = torch.deg2rad(alpha_deg)
    return torch.stack([
        (fz * a.cos() - fx * a.sin()) / qs,  # C_L: along (-sin a, 0, cos a)
        (fx * a.cos() + fz * a.sin()) / qs,  # C_D: along the freestream (cos a, 0, sin a)
        fy / qs,  # C_Y: to starboard
        -mx / (qs * rig.span),  # C_l
        my / (qs * rig.mac),  # C_m
        -mz / (qs * rig.span),  # C_n
    ], dim=-1)


class WindTunnelEnv(Env):
    priv_dim = len(LOADS)
    priv_names: tuple[str, ...] = LOADS

    def __init__(self, spec: EnvSpec, params: WindTunnelParams, n_envs: int = 1, seed: int | None = None) -> None:
        if n_envs != 1:
            raise ValueError("There is one wind tunnel: WindTunnelEnv needs n_envs = 1")
        super().__init__(spec, n_envs, seed)
        self.params = params
        n_taps, n_surfaces = params.rig.n_taps, params.rig.n_surfaces
        self.tracked = params.tracked
        # [targets, alpha, q, taps, commands]: the session reads taps and commands off the end
        self.obs_dim = len(self.tracked) + 2 + n_taps + n_surfaces
        self.act_dim = n_surfaces
        targets = [params.tracking[name] for name in self.tracked]
        self._tracked_cols = torch.tensor([LOADS.index(name) for name in self.tracked])
        self._scale = torch.tensor([t.scale for t in targets])
        self._weight = torch.tensor([t.weight for t in targets])
        self._low = torch.tensor([t.target[0] for t in targets])
        self._high = torch.tensor([t.target[1] for t in targets])
        self._load_scale = torch.tensor([LOAD_SCALES[name] for name in LOADS])
        self._load_cols = torch.tensor([COEFFICIENTS.index(name) for name in LOADS])  # LOADS among COEFFICIENTS
        self._retarget_steps = (None if params.retarget_every_s is None
                                else max(1, round(params.retarget_every_s / spec.dt)))

        # Hardware is opened on the first reset, not here: building the env just to read its dimensions (e.g. in
        # make_policy) must never start the rig.
        self.io: TunnelIO | None = None
        self.clock = FixedRateClock(spec.dt)
        self.safety = SafetyLayer(params.safety, n_surfaces, spec.dt, neutral=params.rig.neutral_action)

        self.taps = torch.zeros(n_taps)  # Cp of the last tick
        self.q = torch.tensor(Q_SCALE_PA)  # dynamic pressure [Pa] of the last tick
        self.alpha = torch.tensor(0.0)  # angle of attack [deg] of the last tick
        self.coeffs = torch.zeros(len(LOADS))  # LOADS of the last tick
        self.target = self._low.clone()  # of the tracked loads, now
        self._pinned: dict[int, float] = {}  # index in `tracked` -> target set by set_targets, kept over redraws
        self._last_error = torch.zeros(len(self.tracked))  # tracked load - its target, over its scale, last step
        self.u = self._neutral()  # command in effect (after safety), normalised
        self.samples = 0  # DAQ samples averaged into the last tick

        # Totals over the current episode, for episode_info.
        self._steps = 0
        self._limited = 0  # surface commands (one per surface per step) the safety layer changed
        self._gap = 0.0  # summed |applied - requested| over those commands, in action units
        self._late_max = 0.0  # worst tick lateness [s]
        self._samples_min: int | None = None
        self._q_sum = 0.0  # summed per-tick q and alpha, for the episode's means
        self._alpha_sum = 0.0

    def _open(self) -> TunnelIO:
        match self.params.io:
            case "mock":
                return MockTunnelIO(self.params.rig, self.params.mock)

    def _read(self) -> bool:
        """Average the samples since the last tick into one reading. Keeps the old one if none arrived.

        False if the new reading is not finite, or q is below MIN_Q_PA (a sensor or DAQ fault, or no flow). The old one
        is kept then too, so nothing non-finite reaches the policy or the replay buffer; the caller trips the safety
        layer."""
        if self.io is None:
            return True
        block = self.io.read_block()
        self.samples = block.t.numel()
        if self.samples == 0:
            return True
        taps, loads, q, alpha = block.taps.mean(0), block.loads.mean(0), block.q.mean(), block.alpha.mean()
        readings = torch.cat([taps, loads, q.reshape(1), alpha.reshape(1)])
        if not torch.isfinite(readings).all() or q < MIN_Q_PA:
            return False
        self.q, self.alpha = q, alpha
        self.taps = taps / q
        self.coeffs = load_coefficients(loads, q, alpha, self.params.rig)[self._load_cols]
        return True

    @property
    def cm(self) -> Tensor:
        return self.coeffs[LOADS.index("C_m")]

    def _draw_targets(self) -> None:
        """A new target for every tracked load, uniform in its range, except those pinned by set_targets."""
        self.target = self._low + (self._high - self._low) * torch.rand(len(self.tracked), generator=self.rng)
        for k, value in self._pinned.items():
            self.target[k] = value

    def set_targets(self, targets: Mapping[str, float]) -> None:
        """Pin the targets of these tracked loads, e.g. {"C_L": 0.45, "C_m": 0.0}, from now on (mid-episode too: the
        next observation shows them) until set again; the others are drawn as in training. {} unpins them all.
        Each must be inside the range it was trained on (check_targets)."""
        check_targets(targets, self.params.tracking)
        self._pinned = {self.tracked.index(name): float(value) for name, value in targets.items()}
        for k, value in self._pinned.items():
            self.target[k] = value

    @property
    def eval_targets(self) -> dict[str, float]:
        return dict(self.params.eval_targets)

    def _neutral(self) -> Tensor:
        """The normalised command that holds every surface at 0 deg."""
        return self.params.rig.neutral_action

    def zero_action(self, obs: Tensor) -> Tensor:
        """Uncontrolled baseline: every surface at neutral, which is not action 0 when the travel is one-sided."""
        return self._neutral().expand(obs.shape[0], -1).clone()

    def _obs(self) -> Tensor:
        # Each target as where it sits in its range, -1..1 (0 for a fixed target), so it is O(1) whatever the load.
        half = (self._high - self._low) / 2
        target = torch.where(half > 0, (self.target - (self._low + half)) / half.clamp_min(1e-12), 0.0)
        flow = torch.stack([self.alpha / ALPHA_SCALE_DEG, self.q / Q_SCALE_PA])
        return torch.cat([target, flow, self.taps, self.u]).unsqueeze(0)

    def reset(self) -> Tensor:
        if self.io is None:
            self.io = self._open()
        self.t = 0
        self._steps, self._limited, self._gap, self._late_max, self._samples_min = 0, 0, 0.0, 0.0, None
        self._q_sum, self._alpha_sum = 0.0, 0.0
        self.safety.reset()
        self.u = self._neutral()
        self._draw_targets()
        self._last_error = torch.zeros(len(self.tracked))
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
        self._q_sum += float(self.q)
        self._alpha_sum += float(self.alpha)

        # The reward is for the command that was applied during this step, against the targets in force during it,
        # whatever happens to either next. Effort and rate are counted in deflection from neutral, so holding the
        # surfaces flush costs nothing.
        deflection, prev_deflection = p.rig.deflection_fraction(applied), p.rig.deflection_fraction(prev_u)
        error = (self.coeffs[self._tracked_cols] - self.target) / self._scale
        reward = tracking_reward(
            error.unsqueeze(0), self._weight, deflection.unsqueeze(0), prev_deflection.unsqueeze(0),
            p.action_cost, p.rate_cost,
        )
        self._last_error = error
        if self.safety.tripped:
            self.io.safe_state()
            self.u = self._neutral()  # the command now in effect, which the next observation reports
            if not was_tripped:
                reward = reward - p.trip_penalty
        if self._retarget_steps is not None and self.t % self._retarget_steps == 0:
            self._draw_targets()  # a step change, seen in the next observation
        return self._obs(), reward

    def privileged(self) -> Tensor:
        return (self.coeffs / self._load_scale).unsqueeze(0)

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
            # The operating point it ran at, which drifts as the tunnel runs.
            "q_pa_mean": self._q_sum / max(self._steps, 1),
            "alpha_deg_mean": self._alpha_sum / max(self._steps, 1),
            # The targets at the end of the episode, and how many were pinned (set_targets) rather than drawn.
            **{f"target_{name}": float(self.target[k]) for k, name in enumerate(self.tracked)},
            "targets_pinned": float(len(self._pinned)),
        }

    def extras(self) -> dict[str, Tensor]:
        """Per step, as measured during it: Cm (always, for the session view and the safety history), and each
        tracked load with the target it was held to (after a step, so a retarget shows on the next one)."""
        out = {
            "cm": self.cm.unsqueeze(0),  # raw Cm, as measured: the privileged copy is scaled for the critic
            "late_ms": torch.tensor([1000.0 * self.clock.last_lateness]),
            "samples": torch.tensor([float(self.samples)]),
            "tripped": torch.tensor([float(self.safety.tripped)]),
        }
        error = self._last_error * self._scale  # back in coefficient units
        for k, name in enumerate(self.tracked):
            out[name] = self.coeffs[LOADS.index(name)].unsqueeze(0)
            out[f"{name}_error"] = error[k].unsqueeze(0)
        return out

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        cm = trace.extras["cm"]
        metrics = {
            "cm_rms": float(cm.pow(2).mean().sqrt()),
            "cm_peak": float(cm.abs().max()),
            "late_ms_max": float(trace.extras["late_ms"].max()),
            "missed_ticks": float(self.clock.misses),
        }
        for name in self.tracked:  # how far from its target each tracked load was
            error = trace.extras[f"{name}_error"]
            metrics[f"{name}_error_rms"] = float(error.pow(2).mean().sqrt())
            metrics[f"{name}_error_peak"] = float(error.abs().max())
        return metrics

    def close(self) -> None:
        if self.io is not None:
            self.io.close()
            self.io = None
        self.clock.close()
