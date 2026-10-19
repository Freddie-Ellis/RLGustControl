"""A CFD case file: everything needed to run one case, so a job takes the case file as its only argument.

Case files are TOML, kept in configs/cases/, and the file name (without .toml) is the case name. Example:

    description = "Both wingtips at +10 deg"
    geometry = "cad/uav/geometry.toml"     # model: bodies, hinges, limits
    mesh = "configs/mesh/uav_test.toml"    # mesh settings
    [deflections_deg]                      # control surfaces not listed stay at 0
    wingtip_r = 10.0
    wingtip_l = 10.0

Paths are relative to the repository root, which is the working directory the HPC jobs run in.
"""

import tomllib
from pathlib import Path

from src.cfd.geometry import GeometryConfig, StrictModel


class CaseSpec(StrictModel):
    description: str = ""
    geometry: Path
    mesh: Path
    deflections_deg: dict[str, float] = {}

    @classmethod
    def from_toml(cls, path: Path) -> CaseSpec:
        """Load a case file and check its deflections against the geometry (names and limits)."""
        with open(path, "rb") as f:
            case = cls.model_validate(tomllib.load(f))
        for file in (case.geometry, case.mesh):
            if not file.is_file():
                raise FileNotFoundError(f"{path}: {file} not found (paths are relative to the repo root)")
        case.load_geometry().check_deflections(case.deflections_deg)
        return case

    def load_geometry(self) -> GeometryConfig:
        return GeometryConfig.from_toml(self.geometry)
