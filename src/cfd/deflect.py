"""Deflect control surfaces about their hinges and write the geometry for one CFD case.

Pipeline for a case, given a deflection angle for some or all control surfaces:
    1. Load every body's STL (as exported from SolidWorks at zero deflection) and fill any small holes.
    2. Rotate each control surface about its hinge axis by its deflection (unlisted surfaces stay at 0).
    3. Convert to metres and write one STL file containing every body as its own named solid.

Each body stays a separate solid, so it becomes its own patch in the OpenFOAM mesh, and forces and hinge
moments can be read off per control surface. Bodies are not merged into one surface: where a deflected
surface overlaps a fixed one, snappyHexMesh simply keeps the fluid outside both.

Run as a script to write the geometry and inspect it in a viewer, e.g.:
    uv run python -m src.cfd.deflect cad/uav/geometry.toml out/aircraft.stl -d wingtip_r=10 wingtip_l=10 --cad-units
"""

import argparse
import math
from pathlib import Path

import torch

from src.cfd.geometry import Body, GeometryConfig, HingeAxis
from src.cfd.stl import Triangles, read_stl, write_ascii_stl
from src.cfd.surface import close_surface


def rotate_about_axis(
    points: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor, angle_rad: float
) -> torch.Tensor:
    """Rotate points (any shape ending in 3) about the line through `origin` along unit vector `direction`.

    Positive angles turn anticlockwise when looking back along `direction` (right-hand rule). Uses Rodrigues'
    rotation formula, applied to each point's position v relative to the axis origin:
        v_rot = v cos(a) + (k x v) sin(a) + k (k . v)(1 - cos(a))
    """
    k = direction
    v = points - origin
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    k_cross_v = torch.linalg.cross(k.expand_as(v), v, dim=-1)
    k_dot_v = (v * k).sum(dim=-1, keepdim=True)
    return origin + v * cos + k_cross_v * sin + k * k_dot_v * (1.0 - cos)


def deflect(triangles: Triangles, hinge: HingeAxis, deflection_deg: float) -> Triangles:
    """Rotate a control surface about its hinge. Positive deflection is the right-hand rotation about p1 -> p2."""
    origin = torch.tensor(hinge.p1, dtype=triangles.dtype)
    direction = torch.tensor(hinge.direction, dtype=triangles.dtype)
    return rotate_about_axis(triangles, origin, direction, math.radians(deflection_deg))


def load_body(geometry: GeometryConfig, body: Body) -> Triangles:
    """Read a body's STL in model coordinates (the frame the hinge points are given in) and close any holes."""
    triangles = read_stl(geometry.stl_path(body)) - torch.tensor(geometry.stl_offset, dtype=torch.float64)
    try:
        return close_surface(triangles, geometry.max_hole_perimeter)
    except ValueError as e:
        raise ValueError(f"{body.name} ({body.stl}): {e}") from e


def deflected_bodies(
    geometry: GeometryConfig, deflections_deg: dict[str, float], in_metres: bool = True
) -> dict[str, Triangles]:
    """Every body of the model, keyed by name, with control surfaces at the requested deflections.

    `deflections_deg` maps control surface names to angles; surfaces not listed stay at 0. Raises if a name
    is unknown or an angle is outside that surface's limits. Output is in metres unless `in_metres` is False,
    in which case it stays in the model's units (useful for overlaying on the CAD model).
    """
    geometry.check_deflections(deflections_deg)

    bodies: dict[str, Triangles] = {}
    for body in geometry.fixed_bodies:
        bodies[body.name] = load_body(geometry, body)
    for surface in geometry.control_surfaces:
        angle = deflections_deg.get(surface.name, 0.0)
        bodies[surface.name] = deflect(load_body(geometry, surface), surface.hinge, angle)

    # Rotate first and scale afterwards, because the hinge points are in the model's units.
    scale = geometry.scale_to_m if in_metres else 1.0
    return {name: triangles * scale for name, triangles in bodies.items()}


def write_case_geometry(geometry: GeometryConfig, deflections_deg: dict[str, float], path: Path) -> None:
    """Write the whole aircraft, in metres, as one STL file with a named solid per body."""
    write_ascii_stl(path, deflected_bodies(geometry, deflections_deg))


def parse_deflections(items: list[str]) -> dict[str, float]:
    """Turn command-line items like ["wingtip_r=10", "wingtip_l=-5"] into {"wingtip_r": 10.0, ...}."""
    deflections: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected NAME=DEG, got {item!r}")
        deflections[name] = float(value)
    return deflections


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write the deflected aircraft as one STL file (a named solid per body), e.g. to view it.",
    )
    parser.add_argument("geometry", type=Path, help="path to geometry.toml")
    parser.add_argument("out_file", type=Path, help="STL file to write")
    parser.add_argument("-d", "--deflect", nargs="*", default=[], metavar="NAME=DEG",
                        help="control surface deflections in degrees; unlisted surfaces stay at 0")
    parser.add_argument("--cad-units", action="store_true",
                        help="write in the model's units (e.g. mm) instead of metres, to overlay on the CAD model")
    args = parser.parse_args(argv)

    deflections = parse_deflections(args.deflect)
    geometry = GeometryConfig.from_toml(args.geometry)
    write_ascii_stl(args.out_file, deflected_bodies(geometry, deflections, in_metres=not args.cad_units))
    print(f"wrote {args.out_file} with deflections {deflections or 'all zero'}")


if __name__ == "__main__":
    main()
