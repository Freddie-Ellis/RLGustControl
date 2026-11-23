"""Thinned CFD cases (paraview_scripts/thin_cases.py) for the CFD tab's 3D viewer: finding and loading them, the fields
computed from them, and the colour maps to draw them with.

thin_cases.py writes, per solved condition:

    <dir>/<batch>_<condition>/walls.vtp                  cell data p (kinematic)
    <dir>/<batch>_<condition>/x_136mm.vtp, y_m118mm.vtp  cell data p, U
    <dir>/<batch>_<condition>/case.json                  freestream_velocity_m_s

Everything is in the model frame (x aft, y starboard, z up): the wing stays still and the freestream turns.

The fields, ranges and colour maps match paraview_scripts/slices.py (keep them in step), so the viewer shows what the
rendered images showed: Cp on the walls; on the slices Cp, |U|, or the disturbance U - U_inf along x, y or z. Signed
fields are blue - white - red with white at exactly 0; |U| uses ParaView's default "Cool to Warm".
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

CFD_VIEW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "cfd_view"  # thin_cases.py's <outdir>, copied
AXES: tuple[str, ...] = ("x", "y")
U_INF: float = 20.0  # m/s: sets the velocity colour ranges, and the freestream of a case with no case.json
CP_RANGE: tuple[float, float] = (-4.0, 1.0)  # as slices.py and wall_cp_animation.py

# (value as a fraction of the range, r, g, b): blue - white - red, white placed at 0 by `colour_table`
_DIVERGING = ((0.23, 0.30, 0.75), (1.0, 1.0, 1.0), (0.71, 0.02, 0.15))  # slices.py DIVERGING, interpolated in Lab
_COOL_TO_WARM = ((0.0, 0.23137254902, 0.298039215686, 0.752941176471),  # ParaView's default map, as on the |U| images
                 (0.5, 0.865, 0.865, 0.865),
                 (1.0, 0.705882352941, 0.0156862745098, 0.149019607843))


@dataclass(frozen=True)
class FieldStyle:
    title: str  # colour bar title
    range: tuple[float, float]
    signed: bool  # blue - white - red with white at 0; otherwise Cool to Warm


FIELDS: dict[str, FieldStyle] = {
    "Cp": FieldStyle("Cp", CP_RANGE, True),
    "U": FieldStyle("|U| (m/s)", (0.0, 1.5 * U_INF), False),
    "Ux": FieldStyle("Ux - U_inf,x (m/s)", (-0.5 * U_INF, 0.5 * U_INF), True),
    "Uy": FieldStyle("Uy - U_inf,y (m/s)", (-0.5 * U_INF, 0.5 * U_INF), True),
    "Uz": FieldStyle("Uz - U_inf,z (m/s)", (-0.5 * U_INF, 0.5 * U_INF), True),
}
WALL_FIELD: str = "Cp"

_CASE = re.compile(r"^(?P<batch>.+?)_+a(?P<alpha>[+-]\d+(?:\.\d+)?)_b(?P<beta>[+-]\d+(?:\.\d+)?)$")
_SLICE = re.compile(r"^(?P<axis>[xyz])_(?P<sign>[mp])?(?P<mm>\d+)mm\.vtp$")


@dataclass(frozen=True)
class Slice:
    axis: str
    position_mm: int
    mesh: pv.PolyData


@dataclass(frozen=True)
class CfdCase:
    name: str
    walls: pv.PolyData
    slices: dict[str, list[Slice]]  # axis -> slices in order: x upstream to downstream, y port to starboard
    freestream: NDArray[np.float64]  # model frame, m/s


def list_cases(data_dir: Path = CFD_VIEW_DIR) -> list[str]:
    """Thinned case folders under `data_dir` (complete ones: they have case.json), in `case_order`."""
    if not data_dir.is_dir():
        return []
    return sorted((d.name for d in data_dir.iterdir() if (d / "case.json").is_file()), key=case_order)


def case_order(name: str) -> tuple[bool, str, bool, float, float]:
    """Batch by batch (one set of deflections each); within a batch, the whole alpha sweep at zero sideslip, then the
    sideslip conditions by alpha, then beta. As wall_cp_animation.py's animation_order. Names that are not
    <batch>_a<alpha>_b<beta> go last, by name."""
    m = _CASE.match(name)
    if m is None:
        return (True, name, False, 0.0, 0.0)
    alpha, beta = float(m["alpha"]), float(m["beta"])
    return (False, m["batch"], beta != 0.0, alpha, beta)


def load_case(case_dir: Path) -> CfdCase:
    info = json.loads((case_dir / "case.json").read_text())
    freestream = np.asarray(info.get("freestream_velocity_m_s", [U_INF, 0.0, 0.0]), dtype=np.float64)
    slices: dict[str, list[Slice]] = {axis: [] for axis in AXES}
    for path in case_dir.glob("*.vtp"):
        m = _SLICE.match(path.name)
        if m is None or m["axis"] not in slices:
            continue
        position = -int(m["mm"]) if m["sign"] == "m" else int(m["mm"])
        slices[m["axis"]].append(Slice(m["axis"], position, pv.PolyData(str(path))))
    for axis_slices in slices.values():
        axis_slices.sort(key=lambda s: s.position_mm)
    return CfdCase(case_dir.name, pv.PolyData(str(case_dir / "walls.vtp")), slices, freestream)


def field_values(case: CfdCase, mesh: pv.PolyData, field: str) -> NDArray[np.float32]:
    """Per-cell values of `field` (a key of FIELDS) on one of the case's meshes."""
    if field == "Cp":
        return np.asarray(mesh.cell_data["p"] / (0.5 * float(case.freestream @ case.freestream)), dtype=np.float32)
    velocity = np.asarray(mesh.cell_data["U"])
    if field == "U":
        return np.linalg.norm(velocity, axis=1).astype(np.float32)
    component = "xyz".index(field[1])
    return (velocity[:, component] - case.freestream[component]).astype(np.float32)


def colour_table(field: str, n: int = 256) -> NDArray[np.float64]:
    """`n` RGB colours (0..1) spanning FIELDS[field].range, made by VTK's colour transfer function exactly as ParaView
    does, so the colours match the rendered images."""
    style = FIELDS[field]
    lo, hi = style.range
    ctf = vtkColorTransferFunction()
    if style.signed and lo < 0.0 < hi:
        ctf.SetColorSpaceToLab()
        for value, rgb in zip((lo, 0.0, hi), _DIVERGING):
            ctf.AddRGBPoint(value, *rgb)
    else:
        ctf.SetColorSpaceToDiverging()
        for fraction, r, g, b in _COOL_TO_WARM:
            ctf.AddRGBPoint(lo + fraction * (hi - lo), r, g, b)
    return np.array([ctf.GetColor(v) for v in np.linspace(lo, hi, n)])
