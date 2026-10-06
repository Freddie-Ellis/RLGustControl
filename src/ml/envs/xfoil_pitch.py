"""Pitch disturbed 2D air foil environment for reinforcement learning.

XFOIL (steady, viscous-inviscid) is driven as a quasi-steady solver: every step the effective angle of attack is
set from the gust and the trailing edge flap from the action. Pressure at the taps is the observation. A first-order
lag (`tau`) stands in for the unsteady response XFOIL cannot capture.
"""

import math
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from src.config import CONFIG
from src.ml.envs.base import Env, EnvSpec, Trace

_CL_RE = re.compile(r"CL\s*=\s*(-?\d+\.\d+)")
_NUM_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


class XFOILPitchProblemParams(BaseModel):
    """Parameters for the pitch disturbed 2D air foil environment."""
    model_config = ConfigDict(extra="forbid")

    tau: float = 0.3  # lag of the unsteady response [s]
    cl_ref: float = 0.5
    gust_len: float = 1.0
    gust_amp_deg: float = 4.0  # max gust induced change in angle of attack [deg]
    base_alpha_deg: float = 4.0  # mean angle of attack [deg]
    act_limit: float = 1.0  # actions are clamped to +-act_limit and mapped to +-flap_max_deg
    flap_max_deg: float = 15.0
    flap_hinge_x: float = 0.75  # hinge location [x/c]
    action_cost: float = 0.01
    naca: str = "0012"
    reynolds: float = 1e6
    xfoil_path: str = r"C:\XFOIL6.99\xfoil.exe"
    timeout_s: float = 30.0


@dataclass
class _Solution:
    """One converged XFOIL operating point."""
    cp: Tensor  # [2 * no_sensors] upper surface then lower surface
    cl: float


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of (xp ascending, fp) at x."""
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * ((x - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0.0, 1.0)


class XFOILPitchProblemEnv(Env):

    no_sensors: int = 10  # taps per surface
    obs_dim = no_sensors * 2 + 2  # [pressures (upper, lower), rate of change, previous action]
    act_dim = 1  # [flap angle]

    def __init__(self, spec: EnvSpec, params: XFOILPitchProblemParams, n_envs: int, seed: int | None = None) -> None:
        super().__init__(spec, n_envs, seed)
        self.params = params
        self.sensor_x = torch.linspace(0.05, 0.95, self.no_sensors)
        self.pressures = torch.zeros((n_envs, 2 * self.no_sensors))
        self.cl = torch.zeros(n_envs)
        self.prev_u = torch.zeros(n_envs)
        self.gust_amp = torch.zeros(n_envs)
        self.gust_t0 = torch.zeros(n_envs)
        self._last: list[_Solution] = []

        self.setupXFOIL(naca=self.params.naca)

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self.rng, device=CONFIG.torch_device)

    def _gust_deg(self) -> Tensor:
        """1-cosine gust induced angle of attack change at the current time, for every env [deg]."""
        s = (self.t * self.spec.dt - self.gust_t0) / self.params.gust_len
        active = (s >= 0.0) & (s <= 1.0)
        shape = 0.5 * (1.0 - torch.cos(2.0 * math.pi * s))
        return torch.where(active, self.gust_amp * shape, torch.zeros_like(s))

    def _obs(self, pressure_dot: Tensor) -> Tensor:
        """Construct the observation vector from the current state."""
        return torch.cat([self.pressures, 0.1 * pressure_dot.unsqueeze(-1), self.prev_u.unsqueeze(-1)], dim=-1)

    def _pressure_lift(self) -> Tensor:
        """Lift proxy from the taps: mean pressure difference between the lower and upper surface [N]."""
        n = self.no_sensors
        return (self.pressures[:, n:] - self.pressures[:, :n]).mean(dim=-1)

    def reset(self) -> Tensor:
        """Reset the environment to an initial state and return the initial observation."""
        n = self.n_envs
        self.t = 0
        self.gust_amp = self.params.gust_amp_deg * (2 * self._rand(n) - 1.0)
        self.gust_t0 = 0.5 + 1.5 * self._rand(n)
        self.prev_u = torch.zeros(n)

        sol = self._solve(self.params.base_alpha_deg, 0.0)
        if sol is None:
            raise RuntimeError("XFOIL did not converge at the base operating point")
        self._last = [sol] * n
        self.pressures = sol.cp.repeat(n, 1)
        self.cl = torch.full((n,), sol.cl)
        return self._obs(torch.zeros(n))

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        u = action.squeeze(-1).clamp(-self.params.act_limit, self.params.act_limit)
        alpha = self.params.base_alpha_deg + self._gust_deg()
        flap = u * (self.params.flap_max_deg / self.params.act_limit)

        steady = self.runXFOIL(alpha, flap)
        target_cp = torch.stack([s.cp for s in steady])
        target_cl = torch.tensor([s.cl for s in steady])

        lag = self.spec.dt / self.params.tau
        lift_before = self._pressure_lift()
        self.pressures = self.pressures + lag * (target_cp - self.pressures)
        self.cl = self.cl + lag * (target_cl - self.cl)
        pressure_dot = (self._pressure_lift() - lift_before) / self.spec.dt

        self.prev_u = u
        self.t += 1
        err = self.cl - self.params.cl_ref
        reward = -(err**2) - self.params.action_cost * u**2
        return self._obs(pressure_dot), reward

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        err = trace.extras["cl"] - self.params.cl_ref
        return {"cl_err_rms": float(err.pow(2).mean().sqrt()), "cl_err_peak": float(err.abs().max())}

    def _extras(self) -> dict[str, Tensor]:
        return {"cl": self.cl, "cl_ref": torch.full_like(self.cl, self.params.cl_ref), "gust": self._gust_deg()}

    def setupXFOIL(self, naca: str) -> None:
        """Check XFOIL is available and the foil is valid."""
        if not Path(self.params.xfoil_path).is_file():
            raise FileNotFoundError(f"XFOIL executable not found at {self.params.xfoil_path}")
        self.naca = self.parseNACA(naca)

    def parseNACA(self, naca: str) -> str:
        """Validate a 4 digit NACA code."""
        if not re.fullmatch(r"\d{4}", naca):
            raise ValueError(f"Only 4 digit NACA foils are supported, got {naca!r}")
        return naca

    def runXFOIL(self, alpha_deg: Tensor, flap_deg: Tensor) -> list[_Solution]:
        """Solve every env in parallel. A point that fails to converge keeps its previous solution."""
        with ThreadPoolExecutor(max_workers=self.n_envs) as pool:
            results = list(pool.map(self._solve, alpha_deg.tolist(), flap_deg.tolist()))
        self._last = [new or old for new, old in zip(results, self._last)]
        return self._last

    def _script(self, alpha_deg: float, flap_deg: float, cp_file: str) -> str:
        p = self.params
        flap = f"GDES\nFLAP\n{p.flap_hinge_x}\n0.0\n{flap_deg}\nEXEC\n\n" if abs(flap_deg) > 1e-6 else ""
        return (
            f"PLOP\nG F\n\nNACA {self.naca}\n{flap}PANE\nOPER\nVISC {p.reynolds:g}\nITER 200\n"
            f"ALFA {alpha_deg}\nCPWR {cp_file}\n\nQUIT\n"
        )

    def _solve(self, alpha_deg: float, flap_deg: float) -> _Solution | None:
        """Run XFOIL at one operating point. Returns None if it failed to converge."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                out = subprocess.run(
                    [self.params.xfoil_path], input=self._script(alpha_deg, flap_deg, "cp.txt"),
                    cwd=tmp, capture_output=True, text=True, timeout=self.params.timeout_s, check=False,
                ).stdout
            except subprocess.TimeoutExpired:
                return None
            cl_matches = _CL_RE.findall(out)
            cp_path = Path(tmp) / "cp.txt"
            if "VISCAL:  Convergence failed" in out or not cl_matches or not cp_path.is_file():
                return None
            rows = [
                [float(v) for v in parts]
                for parts in (line.split() for line in cp_path.read_text().splitlines())
                if len(parts) == 3 and all(_NUM_RE.fullmatch(v) for v in parts)
            ]
        xy_cp = torch.tensor(rows)
        x, cp = xy_cp[:, 0].contiguous(), xy_cp[:, 2].contiguous()
        le = int(x.argmin())  # XFOIL orders the surface upper TE -> LE -> lower TE
        upper = _interp(self.sensor_x, x[: le + 1].flip(0).contiguous(), cp[: le + 1].flip(0))
        lower = _interp(self.sensor_x, x[le:], cp[le:])
        return _Solution(cp=torch.cat([upper, lower]), cl=float(cl_matches[-1]))
