"""Thinned CFD cases (paraview_scripts/thin_cases.py) for the CFD tab's 3D viewer: finding and loading them, the fields
computed from them, and the colour maps to draw them with.

thin_cases.py writes, per solved condition:

    <dir>/<batch>_<condition>/walls.vtp                  cell data p (kinematic)
    <dir>/<batch>_<condition>/x_136mm.vtp, y_m118mm.vtp  cell data p, U
    <dir>/<batch>_<condition>/case.json                  freestream_velocity_m_s

Cases can also sit one folder down, one folder per set (e.g. <dir>/actuator_effects/...). A set with a
setpoints.parquet (the cases' deflections, one row per <case>_<condition> folder) moves one actuator at a time and is
viewed as actuator sweeps (`actuator_sweeps`).

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
import polars as pl
import pyvista as pv
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from pyvista.plotting._typing import ScalarBarArgs
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

CFD_VIEW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "cfd_view"  # thin_cases.py's <outdir>, copied
SETPOINTS: str = "setpoints.parquet"  # a set's per-case deflections, which make it viewable as actuator sweeps
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

# How a case is drawn, in the CFD tab and the GIFs alike
VIEW_DIRECTION: tuple[float, float, float] = (-0.8, -1.0, 0.6)  # slices.py's "iso" camera
SLICE_LIGHTING: tuple[float, float] = (1.0, 0.0)  # (ambient, diffuse) as slices.py: flat slices in true colours
WALL_LIGHTING: tuple[float, float] = (0.6, 0.4)  # mostly unlit so white reads as white, a little shading for shape
BAR_LABELS: dict[str, int] = {"Cp": 6, "U": 7, "Ux": 5, "Uy": 5, "Uz": 5}  # labels at round steps of each range

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


def list_sets(data_dir: Path = CFD_VIEW_DIR) -> list[str]:
    """Sets of cases: "" for the cases directly in `data_dir`, then each subfolder holding cases (e.g.
    actuator_effects), by name."""
    if not data_dir.is_dir():
        return []
    subsets = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and list_cases(d))
    return ([""] if list_cases(data_dir) else []) + subsets


@dataclass(frozen=True)
class SweepPoint:
    deflection_deg: float
    case: str


type Sweeps = dict[str, dict[tuple[float, float], list[SweepPoint]]]  # actuator -> (alpha, beta) -> by deflection


def actuator_sweeps(set_dir: Path) -> Sweeps:
    """The set's actuator sweeps, from its SETPOINTS file (one row per solved case and condition): an actuator is a
    `family` moved in one `mode` ("wingtip (sym)", "le (right)", ...) by `delta_deg`; its sweep at each flow condition
    is its cases there, by deflection, with the "clean" family's case there as the 0 deg point. Rows with no
    delta_deg (the "combined" case) fit no sweep. Empty if the set has no SETPOINTS file."""
    path = set_dir / SETPOINTS
    if not path.is_file():
        return {}
    available = set(list_cases(set_dir))
    rows = (pl.read_parquet(path, columns=["case", "condition", "family", "mode", "delta_deg", "alpha_deg", "beta_deg"])
            .with_columns(name=pl.col("case") + "_" + pl.col("condition"))
            .filter(pl.col("name").is_in(available)))
    clean = {(row["alpha_deg"], row["beta_deg"]): row["name"]
             for row in rows.filter(pl.col("family") == "clean").iter_rows(named=True)}
    moved_alone: dict[str, dict[tuple[float, float], list[SweepPoint]]] = {}
    for row in rows.filter(pl.col("delta_deg").is_not_null()).iter_rows(named=True):
        actuator = f"{row['family']} ({row['mode']})"
        condition = (row["alpha_deg"], row["beta_deg"])
        moved_alone.setdefault(actuator, {}).setdefault(condition, []).append(SweepPoint(row["delta_deg"], row["name"]))
    sweeps: Sweeps = {}
    for actuator in sorted(moved_alone):
        sweeps[actuator] = {}
        for condition in sorted(moved_alone[actuator], key=lambda c: (c[1] != 0.0, c[0], c[1])):
            zero = [SweepPoint(0.0, clean[condition])] if condition in clean else []
            sweeps[actuator][condition] = sorted(moved_alone[actuator][condition] + zero, key=lambda p: p.deflection_deg)
    return sweeps


def case_condition(name: str) -> tuple[str, float, float] | None:
    """(batch, alpha, beta) of a <batch>_a<alpha>_b<beta> case name, or None if it is not one."""
    m = _CASE.match(name)
    return (m["batch"], float(m["alpha"]), float(m["beta"])) if m else None


def case_order(name: str) -> tuple[bool, str, bool, float, float]:
    """Batch by batch (one set of deflections each); within a batch, the whole alpha sweep at zero sideslip, then the
    sideslip conditions by alpha, then beta. As wall_cp_animation.py's animation_order. Names that are not
    <batch>_a<alpha>_b<beta> go last, by name."""
    m = _CASE.match(name)
    if m is None:
        return (True, name, False, 0.0, 0.0)
    alpha, beta = float(m["alpha"]), float(m["beta"])
    return (False, m["batch"], beta != 0.0, alpha, beta)


def load_case(case_dir: Path, with_slices: bool = True) -> CfdCase:
    """The case in `case_dir`; without its slices (none listed) if not `with_slices`, which is much quicker."""
    info = json.loads((case_dir / "case.json").read_text())
    freestream = np.asarray(info.get("freestream_velocity_m_s", [U_INF, 0.0, 0.0]), dtype=np.float64)
    slices: dict[str, list[Slice]] = {axis: [] for axis in AXES}
    for path in case_dir.glob("*.vtp") if with_slices else []:
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


def bar_args(field: str, title: str, position_y: float) -> ScalarBarArgs:
    """A vertical colour bar on the right, black text on the white background (as the rendered images)."""
    return {"title": title, "n_labels": BAR_LABELS.get(field, 5), "position_x": 0.9, "position_y": position_y,
            "vertical": True, "color": "black", "fmt": "%.3g", "width": 0.05, "height": 0.35,
            "title_font_size": 14, "label_font_size": 12}


def add_field(plotter: pv.BasePlotter, case: CfdCase, mesh: pv.PolyData, field: str, lighting: tuple[float, float],
              bar: ScalarBarArgs | None) -> pv.Actor:
    """Draw `mesh` coloured cell by cell by `field` in its fixed colour map and range, with (ambient, diffuse)
    `lighting`, and a colour bar if `bar` is given. Leaves the camera alone."""
    ambient, diffuse = lighting
    return plotter.add_mesh(
        mesh, scalars=field_values(case, mesh, field), preference="cell",
        cmap=ListedColormap(colour_table(field)), clim=FIELDS[field].range,
        ambient=ambient, diffuse=diffuse, specular=0.0, reset_camera=False,
        show_scalar_bar=bar is not None, scalar_bar_args=bar,
    )


def iso_camera(meshes: list[pv.PolyData], zoom: float = 1.0,
               direction: tuple[float, float, float] = VIEW_DIRECTION) -> list[tuple[float, float, float]]:
    """A camera position framing `meshes`, looking from `direction` (camera side of the centre), z up; `zoom` > 1
    moves in."""
    bounds = np.array([m.bounds for m in meshes])
    lo, hi = bounds[:, 0::2].min(axis=0), bounds[:, 1::2].max(axis=0)
    centre = (lo + hi) / 2
    unit = np.array(direction) / np.linalg.norm(direction)
    distance = 1.5 * float(np.linalg.norm(hi - lo)) / zoom
    up = (-1.0, 0.0, 0.0) if abs(unit[2]) > 0.99 else (0.0, 0.0, 1.0)  # looking straight down: nose to the top
    return [tuple(centre + distance * unit), tuple(centre), up]


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
