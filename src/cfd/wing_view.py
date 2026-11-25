"""The wing as the Session tab's replay draws it: the CAD model's bodies as meshes, where each control surface sits at
a given deflection, and where the pressure taps are. No Qt here; the replay panel only draws what this returns.

The bodies are loaded once, undeflected. A surface is moved to its deflection by a 4 x 4 transform (`surface_matrix`),
the same rotation about its hinge as `deflect.deflect` (right hand about p1 -> p2), so a frame of an animation only
swaps 12 matrices instead of rebuilding meshes.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyvista as pv
from numpy.typing import NDArray

from src.cfd.batch_cases import ACTUATOR_SURFACES
from src.cfd.deflect import load_body
from src.cfd.geometry import Body, GeometryConfig
from src.config import CONFIG
from src.ml.taps.cfd_taps import OUTPUT_DIR as TAP_SELECTIONS_DIR

WING_GEOMETRY: Path = CONFIG.configs_dir.parent / "cad" / "uav" / "geometry.toml"
# TODO: confirm the rig's taps are wired in this layout's order (DAQ channel i = the layout's tap i).
TAPS_FILE: Path = TAP_SELECTIONS_DIR / "taps.parquet"


@dataclass(frozen=True)
class Hinge:
    origin: NDArray[np.float64]  # a point on the hinge line [m]
    direction: NDArray[np.float64]  # unit vector along it, p1 -> p2


@dataclass(frozen=True)
class WingModel:
    fixed: dict[str, pv.PolyData]  # bodies that never move, in metres, model frame (x aft, y starboard, z up)
    surfaces: dict[str, pv.PolyData]  # control surfaces at 0 deg, in metres
    hinges: dict[str, Hinge]
    rig_order: tuple[str, ...]  # the control surface each rig actuator k moves (CONFIG.rig order)

    def surface_matrix(self, name: str, deflection_deg: float) -> NDArray[np.float64]:
        """The 4 x 4 transform that takes surface `name` from 0 deg to `deflection_deg` about its hinge."""
        hinge = self.hinges[name]
        k = hinge.direction
        a = math.radians(deflection_deg)
        cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        rotation = np.eye(3) * math.cos(a) + cross * math.sin(a) + np.outer(k, k) * (1.0 - math.cos(a))  # Rodrigues
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = hinge.origin - rotation @ hinge.origin
        return matrix


def _mesh(triangles: NDArray[np.float64]) -> pv.PolyData:
    """Triangles (n, 3, 3) -> a surface mesh, corners merged so it shades smoothly."""
    points = triangles.reshape(-1, 3)
    faces = np.hstack([np.full((len(triangles), 1), 3), np.arange(len(points)).reshape(-1, 3)]).ravel()
    return pv.PolyData(points, faces).clean()


def load_wing_model(path: Path = WING_GEOMETRY) -> WingModel:
    geometry = GeometryConfig.from_toml(path)
    scale = geometry.scale_to_m

    def body_mesh(body: Body) -> pv.PolyData:
        return _mesh(load_body(geometry, body).numpy() * scale)

    hinges = {}
    for surface in geometry.control_surfaces:
        p1, p2 = np.array(surface.hinge.p1) * scale, np.array(surface.hinge.p2) * scale
        hinges[surface.name] = Hinge(origin=p1, direction=(p2 - p1) / np.linalg.norm(p2 - p1))
    if len(ACTUATOR_SURFACES) != CONFIG.rig.n_surfaces or set(ACTUATOR_SURFACES) - set(hinges):
        raise ValueError(f"the rig's {CONFIG.rig.n_surfaces} surfaces {ACTUATOR_SURFACES} do not match the geometry's "
                         f"{sorted(hinges)}")
    return WingModel(
        fixed={b.name: body_mesh(b) for b in geometry.fixed_bodies},
        surfaces={s.name: body_mesh(s) for s in geometry.control_surfaces},
        hinges=hinges,
        rig_order=ACTUATOR_SURFACES,
    )


def tap_positions(path: Path = TAPS_FILE) -> NDArray[np.float64]:
    """Where each rig tap is, in metres (n_taps, 3), in the order the DAQ reads them (see TAPS_FILE)."""
    return pl.read_parquet(path).sort("order").select("x", "y", "z").to_numpy().astype(np.float64)
