"""Description of a CAD model for the CFD pipeline, loaded from `cad/<model>/geometry.toml`.

The model is exported from SolidWorks at zero deflection as one STL file per body. Bodies come in two kinds:
    fixed bodies       used as exported (e.g. the main body);
    control surfaces   rotated about their hinge axis to each case's deflection (see `deflect.py`).

Conventions (all coordinates are in the model's frame and `units`, the same as the SolidWorks part):
    x streamwise (nose -> tail), y spanwise (to starboard), z up.
    Positive deflection is a right-hand rotation about the hinge axis directed from `p1` to `p2`: point your
    right thumb from p1 to p2, and your fingers curl in the positive direction. Swapping p1 and p2 reverses
    the sign.

Every model is a Pydantic model, so the TOML is checked on load: unknown keys, missing values, bad names,
duplicate bodies and inconsistent limits are all reported as errors.
"""

import math
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Vec3 = tuple[float, float, float]

# Body names become file names and OpenFOAM patch names, so keep them to simple lowercase identifiers.
BODY_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

# Multiply a length in these units by the factor to get metres.
UNIT_SCALE_TO_M: dict[str, float] = {"m": 1.0, "mm": 1e-3}


class StrictModel(BaseModel):
    """Base for every config model: unknown keys are errors (catches typos), and values are read-only."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class HingeAxis(StrictModel):
    """The line a control surface rotates about, given by two points on it.

    The order of the points sets the sign of the deflection (see the module docstring).
    """
    p1: Vec3
    p2: Vec3

    @model_validator(mode="after")
    def _distinct_points(self) -> Self:
        if math.dist(self.p1, self.p2) == 0.0:
            raise ValueError("hinge axis points p1 and p2 must be distinct")
        return self

    @property
    def direction(self) -> Vec3:
        """Unit vector pointing from p1 to p2."""
        length = math.dist(self.p1, self.p2)
        dx, dy, dz = (b - a for a, b in zip(self.p1, self.p2))
        return (dx / length, dy / length, dz / length)


class Body(StrictModel):
    """One closed surface of the model, exported from SolidWorks as its own STL file."""
    name: str = Field(pattern=BODY_NAME_PATTERN)  # becomes the OpenFOAM patch name
    stl: Path                                     # relative to the folder containing geometry.toml


class ControlSurface(Body):
    """A body that rotates about a hinge. Its STL is exported at zero deflection."""
    hinge: HingeAxis
    min_deflection_deg: float
    max_deflection_deg: float

    @model_validator(mode="after")
    def _limits_contain_zero(self) -> Self:
        # The STL is the zero-deflection shape, so 0 must be an allowed angle.
        if not self.min_deflection_deg <= 0.0 <= self.max_deflection_deg:
            raise ValueError(
                f"{self.name}: deflection limits [{self.min_deflection_deg}, {self.max_deflection_deg}] must contain 0"
            )
        return self


class Reference(StrictModel):
    """Reference values used to turn forces and moments into coefficients (C_L, C_m, ...)."""
    point: Vec3                  # moment reference point, e.g. the centre of gravity
    length: float = Field(gt=0)  # reference length, e.g. the mean aerodynamic chord
    area: float = Field(gt=0)    # reference area, e.g. the wing planform area, in units^2


class GeometryConfig(StrictModel):
    """The whole model: its bodies, units and reference values. Load it with `GeometryConfig.from_toml`."""
    name: str
    units: Literal["m", "mm"] = "m"  # units of the STL files and of every coordinate in the TOML
    reference: Reference

    # Shift between the STL files and the model frame: STL coordinates = model coordinates + stl_offset.
    # Only needed if SolidWorks exported with "translate STL output data to positive space"; subtracted on
    # load so that the hinge points, which are measured in the model frame, still line up.
    stl_offset: Vec3 = (0.0, 0.0, 0.0)

    # Holes in an STL with a perimeter up to this length (in `units`) are filled on load; any larger hole is
    # an error. 0 means every STL must already be closed. See `surface.close_surface`.
    max_hole_perimeter: float = Field(default=0.0, ge=0)

    fixed_bodies: list[Body] = Field(min_length=1)
    control_surfaces: list[ControlSurface] = []

    # Folder containing the geometry.toml, which STL paths are relative to. Filled in by from_toml.
    root: Path = Path(".")

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [b.name for b in self.bodies]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate body names: {duplicates}")
        return self

    @classmethod
    def from_toml(cls, path: Path) -> GeometryConfig:
        """Load and validate a geometry.toml."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate({**data, "root": path.resolve().parent})

    @property
    def bodies(self) -> list[Body]:
        """Fixed bodies followed by control surfaces."""
        return [*self.fixed_bodies, *self.control_surfaces]

    @property
    def scale_to_m(self) -> float:
        """Factor converting the model's units to metres."""
        return UNIT_SCALE_TO_M[self.units]

    def stl_path(self, body: Body) -> Path:
        return self.root / body.stl

    def missing_stls(self) -> list[Path]:
        """STL files named in the config that don't exist on disk."""
        return [p for p in (self.stl_path(b) for b in self.bodies) if not p.is_file()]

    def control_surface(self, name: str) -> ControlSurface:
        for surface in self.control_surfaces:
            if surface.name == name:
                return surface
        raise KeyError(f"no control surface named {name!r}; have {[s.name for s in self.control_surfaces]}")

    def check_deflections(self, deflections_deg: dict[str, float]) -> None:
        """Raise if a deflection names an unknown control surface or is outside that surface's limits."""
        for name, angle in deflections_deg.items():
            surface = self.control_surface(name)
            if not surface.min_deflection_deg <= angle <= surface.max_deflection_deg:
                raise ValueError(
                    f"{name}: deflection {angle} deg outside limits "
                    f"[{surface.min_deflection_deg}, {surface.max_deflection_deg}]"
                )
