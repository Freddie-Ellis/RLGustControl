"""CFD case design: structured sweeps plus a maximin Latin hypercube over all inputs.

Inputs are the 12 actuator angles, angle of attack and sideslip. The splitter actuators
(`CONFIG.rig.splitter_surfaces`) are either closed or open between `SPLITTER_MIN_OPEN_DEG` and
`splitter_max_deg`, a limit of the CFD mesh (the rig has no dead zone); `SPLITTER_CLOSED_FRACTION`
of the LHS cases have each one closed. Gusts from the active grid
change the effective AoA and sideslip, so steady RANS over these ranges covers the quasi-steady
gust response.

Structured cases (fixed, not optimised):
- AoA sweep with clean actuators: the clean-wing polar, for validation against the tunnel.
- Sideslip sweep at the nominal AoA with clean actuators.
The LHS cases are then spread out around them and cover the actuator effects and their
interaction with AoA and sideslip.
"""

from pathlib import Path

import polars as pl
import torch
from pydantic import BaseModel

from src.config import CONFIG

CASES_DIR: Path = Path(__file__).parent / "cfd_cases"

# The wing itself is defined once, in src/config.py.
ACTUATOR_MIN_DEG: float = CONFIG.rig.actuator_min_deg
ACTUATOR_MAX_DEG: float = CONFIG.rig.actuator_max_deg
AOA_MIN_DEG: float = 0
AOA_MAX_DEG: float = 25
NOMINAL_AOA_DEG: float = 10.0  # placeholder: AoA the wing normally flies at
SIDESLIP_MAX_DEG: float = 10.0  # placeholder: largest sideslip the active grid's gusts produce
ANGLE_RESOLUTION_DEG: float = 0.5  # smallest angle step the CFD setup / mesh deformation can use
# The CFD mesh can't deform a splitter to a small opening: it is either closed or open by at least this much.
SPLITTER_MIN_OPEN_DEG: float = 20.0
SPLITTER_CLOSED_FRACTION: float = 0.2  # share of LHS cases with each splitter closed

AOA_SWEEP_STEP_DEG: float = 1.0
SIDESLIP_SWEEP_STEP_DEG: float = 2.0


class DesignInput(BaseModel, frozen=True):
    """One CFD input: its case-table column, range, resolution and value in structured cases.

    With `open_min` set the input has a dead zone: it is either `lower` (closed) or in
    [`open_min`, `upper`]. The first `closed_fraction` of the unit interval maps to closed and the
    rest spreads over the open range, so the LHS still stratifies the open range.
    """

    name: str
    lower: float
    upper: float
    resolution: float = ANGLE_RESOLUTION_DEG
    neutral: float = 0.0
    open_min: float | None = None
    closed_fraction: float = SPLITTER_CLOSED_FRACTION

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        """Unit-interval LHS coordinate -> value."""
        if self.open_min is None:
            return self.lower + u * (self.upper - self.lower)
        f = self.closed_fraction
        opened = self.open_min + (u - f) / (1.0 - f) * (self.upper - self.open_min)
        return torch.where(u < f, self.lower, opened)

    def to_unit(self, v: torch.Tensor) -> torch.Tensor:
        """Value -> unit-interval coordinate. A closed value sits in the middle of the closed band."""
        if self.open_min is None:
            return (v - self.lower) / (self.upper - self.lower)
        f = self.closed_fraction
        opened = f + (v - self.open_min) / (self.upper - self.open_min) * (1.0 - f)
        return torch.where(v <= self.lower, f / 2, opened.clamp(f, 1.0))


def actuator_name(k: int) -> str:
    return f"actuator_{k}_deg"


def actuator_input(k: int) -> DesignInput:
    rig = CONFIG.rig
    if k in rig.splitter_surfaces:
        return DesignInput(
            name=actuator_name(k),
            lower=float(rig.min_deg[k]),  # closed: flush with the wing
            upper=rig.splitter_max_deg,
            open_min=SPLITTER_MIN_OPEN_DEG,
        )
    return DesignInput(name=actuator_name(k), lower=ACTUATOR_MIN_DEG, upper=ACTUATOR_MAX_DEG)


AOA_NAME: str = "aoa_deg"
SIDESLIP_NAME: str = "sideslip_deg"

DESIGN_INPUTS: tuple[DesignInput, ...] = (
    *(actuator_input(k) for k in range(CONFIG.rig.n_surfaces)),
    DesignInput(name=AOA_NAME, lower=AOA_MIN_DEG, upper=AOA_MAX_DEG, neutral=NOMINAL_AOA_DEG),
    DesignInput(name=SIDESLIP_NAME, lower=-SIDESLIP_MAX_DEG, upper=SIDESLIP_MAX_DEG),
)


def input_names(inputs: tuple[DesignInput, ...] = DESIGN_INPUTS) -> list[str]:
    return [i.name for i in inputs]


def input_bounds(
    inputs: tuple[DesignInput, ...] = DESIGN_INPUTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(lower, upper), each shape (d,)."""
    lower = torch.tensor([i.lower for i in inputs], dtype=torch.float64)
    upper = torch.tensor([i.upper for i in inputs], dtype=torch.float64)
    return lower, upper


def latin_hypercube(n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    """One random LHS in [0, 1]^d with bin-centred points, shape (n, d)."""
    perms = torch.stack([torch.randperm(n, generator=generator) for _ in range(d)], dim=1)
    return (perms.to(torch.float64) + 0.5) / n


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(float("inf"))
    return dist


def maximin_lhs(
    n: int,
    d: int,
    fixed: torch.Tensor | None = None,
    n_restarts: int = 10,
    n_swaps: int = 5000,
    seed: int = 0,
) -> torch.Tensor:
    """LHS in [0, 1]^d optimised for the largest minimum distance between points.

    `fixed` holds points already in the design (in [0, 1]^d, shape (m, d)). They are not
    moved, but distances from the LHS points to them count towards the criterion, so the
    LHS spreads out around them.

    Each restart starts from a random LHS and greedily swaps two entries within one
    column (which preserves the LHS property), keeping swaps that do not reduce the
    minimum distance. Only the two affected rows of the distance matrices are updated.
    """
    gen = torch.Generator().manual_seed(seed)
    if fixed is None:
        fixed = torch.empty((0, d), dtype=torch.float64)

    def score_of(dist: torch.Tensor, fixed_dist: torch.Tensor) -> float:
        score = dist.min().item()
        if fixed_dist.numel():
            score = min(score, fixed_dist.min().item())
        return score

    best = latin_hypercube(n, d, gen)
    best_score = score_of(_pairwise_distances(best), torch.cdist(best, fixed))

    for _ in range(n_restarts):
        x = latin_hypercube(n, d, gen)
        dist = _pairwise_distances(x)
        fixed_dist = torch.cdist(x, fixed)
        score = score_of(dist, fixed_dist)
        for _ in range(n_swaps):
            col = int(torch.randint(d, (1,), generator=gen))
            i, j = torch.randperm(n, generator=gen)[:2].tolist()
            x[[i, j], col] = x[[j, i], col]
            rows = torch.cdist(x[[i, j]], x)
            new_dist = dist.clone()
            new_dist[[i, j], :] = rows
            new_dist[:, [i, j]] = rows.T
            new_dist[i, i] = new_dist[j, j] = float("inf")
            new_fixed_dist = fixed_dist.clone()
            new_fixed_dist[[i, j], :] = torch.cdist(x[[i, j]], fixed)
            new_score = score_of(new_dist, new_fixed_dist)
            if new_score >= score:
                dist, fixed_dist, score = new_dist, new_fixed_dist, new_score
            else:
                x[[i, j], col] = x[[j, i], col]
        if score > best_score:
            best, best_score = x.clone(), score

    return best


def _sweep(lower: float, upper: float, step: float) -> list[float]:
    n = round((upper - lower) / step)
    return [lower + k * step for k in range(n + 1)]


def structured_cases(inputs: tuple[DesignInput, ...] = DESIGN_INPUTS) -> pl.DataFrame:
    """AoA sweep and sideslip sweep, without duplicates.

    Every input not being swept sits at its `neutral` value.
    """
    neutral: dict[str, str | float] = {i.name: i.neutral for i in inputs}
    rows: list[dict[str, str | float]] = []
    for aoa in _sweep(AOA_MIN_DEG, AOA_MAX_DEG, AOA_SWEEP_STEP_DEG):
        rows.append({"kind": "aoa_sweep"} | neutral | {AOA_NAME: aoa})
    for sideslip in _sweep(-SIDESLIP_MAX_DEG, SIDESLIP_MAX_DEG, SIDESLIP_SWEEP_STEP_DEG):
        rows.append({"kind": "sideslip_sweep"} | neutral | {SIDESLIP_NAME: sideslip})
    cases = pl.DataFrame(rows)
    return cases.unique(subset=input_names(inputs), keep="first", maintain_order=True)


def lhs_cases(
    n: int,
    fixed: pl.DataFrame,
    inputs: tuple[DesignInput, ...] = DESIGN_INPUTS,
    seed: int = 0,
) -> pl.DataFrame:
    """`n` maximin LHS cases, spread out around the `fixed` cases already in the design."""
    names = input_names(inputs)
    fixed_values = fixed.select(names).to_torch(dtype=pl.Float64)
    fixed_unit = torch.stack([i.to_unit(fixed_values[:, k]) for k, i in enumerate(inputs)], dim=1)
    unit = maximin_lhs(n, len(inputs), fixed=fixed_unit, seed=seed)
    columns = {
        i.name: (torch.round(i.from_unit(unit[:, k]) / i.resolution) * i.resolution).tolist()
        for k, i in enumerate(inputs)
    }
    return pl.DataFrame({"kind": ["lhs"] * n} | columns)


def cfd_cases(n_lhs: int, existing: pl.DataFrame | None = None, seed: int = 0) -> pl.DataFrame:
    """New CFD cases, numbered on from `existing`.

    First batch (no `existing`): the structured cases plus `n_lhs` LHS cases. Later batches:
    only `n_lhs` new LHS cases, spread out around every case already run.
    """
    if existing is None:
        new = structured_cases()
        new = pl.concat([new, lhs_cases(n_lhs, fixed=new, seed=seed)])
        first_case = 0
    else:
        new = lhs_cases(n_lhs, fixed=existing, seed=seed)
        first_case = int(existing["case"].max()) + 1
    return new.with_row_index("case", offset=first_case)


def main() -> None:
    cases = cfd_cases(n_lhs=120)
    print(cases["kind"].value_counts(sort=True))
    cases.write_csv(CASES_DIR / "cfd_cases1.csv")


if __name__ == "__main__":
    main()
