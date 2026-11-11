"""Deflect control surfaces about their hinges, giving the aircraft's bodies for one CFD case.

Given a deflection angle for some or all control surfaces:
    1. Load every body's STL (as exported from SolidWorks at zero deflection; each must be closed).
    2. Rotate each control surface about its hinge axis by its deflection (unlisted surfaces stay at 0).
    3. Convert to metres.

The bodies are then merged into the aircraft's outer skin for meshing (skin.py), keeping one patch per body
so that forces and hinge moments can be read off per control surface. The mesh job writes the merged skin to
<case>/mesh/constant/triSurface/aircraft.stl, which can be viewed in ParaView.
"""

import math

import torch

from src.cfd.geometry import Body, GeometryConfig, HingeAxis
from src.cfd.stl import Triangles, read_stl


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
    """Read a body's STL in model coordinates (the frame the hinge points are given in)."""
    return read_stl(geometry.stl_path(body)) - torch.tensor(geometry.stl_offset, dtype=torch.float64)


def deflected_bodies(geometry: GeometryConfig, deflections_deg: dict[str, float]) -> dict[str, Triangles]:
    """Every body of the model in metres, keyed by name, with control surfaces at the requested deflections.

    `deflections_deg` maps control surface names to angles; surfaces not listed stay at 0. Raises if a name
    is unknown or an angle is outside that surface's limits.
    """
    geometry.check_deflections(deflections_deg)

    bodies: dict[str, Triangles] = {}
    for body in geometry.fixed_bodies:
        bodies[body.name] = load_body(geometry, body)
    for surface in geometry.control_surfaces:
        angle = deflections_deg.get(surface.name, 0.0)
        bodies[surface.name] = deflect(load_body(geometry, surface), surface.hinge, angle)

    # Rotate first and scale afterwards, because the hinge points are in the model's units.
    return {name: triangles * geometry.scale_to_m for name, triangles in bodies.items()}
