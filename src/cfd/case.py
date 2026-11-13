"""A CFD case file: everything needed to run one case, so a job takes the case file as its only argument.

Case files are TOML, kept in configs/cases/ (subfolders allowed). The file name (without .toml) is the case
name, which names its run folder hpc/runs/<case name>/, so it must be unique across all case files. A case is
one geometry with one set of control surface deflections: it is meshed once, then solved at every flow
condition in [flow] on that mesh. Example:

    description = "Both wingtips at +10 deg"
    geometry = "cad/uav/geometry.toml"     # model: bodies, hinges, limits, reference values
    mesh = "configs/mesh/uav.toml"         # mesh settings
    solver = "configs/solver/uav_rans.toml"  # solver settings; leave out to mesh only
    [deflections_deg]                      # control surfaces not listed stay at 0
    wingtip_r = 10.0
    wingtip_l = 10.0
    [mesh_overrides]                       # optional: change individual mesh settings for this case
    base_cell_size = 0.288
    [solver_overrides]                     # optional: change individual solver settings for this case
    keep_fields = false
    [flow]                                # solved at every combination of these angles
    alpha_deg = [0.0, 10.0, 20.0]
    beta_deg = [0.0, 5.0]
    pairs = [[10.0, -4.0]]                 # optional: extra (alpha, beta) conditions

Paths are relative to the repository root, which is the working directory the HPC jobs run in.
"""

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import model_validator

from src.cfd.geometry import GeometryConfig, StrictModel


class FlowCondition(StrictModel):
    """One flow direction: angle of attack and sideslip, in degrees."""
    alpha_deg: float
    beta_deg: float

    @property
    def name(self) -> str:
        """Folder name for this condition, e.g. a+08.0_b-05.0."""
        return f"a{self.alpha_deg:+05.1f}_b{self.beta_deg:+05.1f}"


class FlowSpec(StrictModel):
    """The flow conditions a case is solved at: every combination of alpha_deg and beta_deg, plus any
    (alpha, beta) pairs listed in `pairs` (e.g. a sideslip sweep at one alpha). Repeats are dropped."""
    alpha_deg: list[float] = []
    beta_deg: list[float] = [0.0]
    pairs: list[tuple[float, float]] = []

    @model_validator(mode="after")
    def _has_conditions(self) -> Self:
        if not self.conditions():
            raise ValueError("[flow] gives no conditions: list alpha_deg, or pairs")
        return self

    def conditions(self) -> list[FlowCondition]:
        angles = [(a, b) for b in self.beta_deg for a in self.alpha_deg] + list(self.pairs)
        return [FlowCondition(alpha_deg=a, beta_deg=b) for a, b in dict.fromkeys(angles)]


class CaseSpec(StrictModel):
    description: str = ""
    geometry: Path
    mesh: Path
    solver: Path | None = None
    deflections_deg: dict[str, float] = {}
    mesh_overrides: dict[str, Any] = {}  # mesh settings that replace those in the `mesh` file
    solver_overrides: dict[str, Any] = {}  # solver settings that replace those in the `solver` file
    flow: FlowSpec | None = None

    @classmethod
    def from_toml(cls, path: Path) -> CaseSpec:
        """Load a case file and check its deflections against the geometry (names and limits)."""
        with open(path, "rb") as f:
            case = cls.model_validate(tomllib.load(f))
        for file in (case.geometry, case.mesh, case.solver):
            if file is not None and not file.is_file():
                raise FileNotFoundError(f"{path}: {file} not found (paths are relative to the repo root)")
        if (case.solver is None) != (case.flow is None):
            raise ValueError(f"{path}: give both `solver` and [flow] to solve the case, or neither to mesh only")
        case.load_geometry().check_deflections(case.deflections_deg)
        return case

    def load_geometry(self) -> GeometryConfig:
        return GeometryConfig.from_toml(self.geometry)

    def mesh_inputs(self) -> dict[str, object]:
        """Everything that determines the mesh. Two case files with the same mesh inputs share a mesh, so
        flow conditions can be added to a case after it has been meshed."""
        return {
            "geometry": str(self.geometry),
            "mesh": str(self.mesh),
            "deflections_deg": {k: v for k, v in sorted(self.deflections_deg.items()) if v != 0.0},
            "mesh_overrides": dict(sorted(self.mesh_overrides.items())),
        }

    def conditions(self) -> list[FlowCondition]:
        """The flow conditions to solve at (none if the case is mesh only)."""
        return self.flow.conditions() if self.flow is not None else []
