"""Shared code for the post-processing scripts: finding run folders, reading results and wall surfaces,
cutting the wall surface at a spanwise station, and printing tables.

A run folder is a downloaded hpc/runs/<case name>/: case.toml, optionally mesh/, and solve/<condition>/, each
with results.json, solve_case.json and postProcessing/. Every script accepts run folders, folders of run
folders (e.g. a whole downloaded batch) and glob patterns.
"""

import argparse
import base64
import glob
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.cfd.solve_report import SolveResult

COEFFICIENTS: tuple[str, ...] = ("C_L", "C_D", "C_Y", "C_l", "C_m", "C_n")
DEFAULT_STATIONS_MM: tuple[float, ...] = (60.0, 120.0, 180.0, 240.0, 300.0)  # spanwise cuts, starboard side
SURFACE_NORMAL_MIN: float = 0.3  # |n_z| above this counts as an upper or lower surface, not a side face


def run_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("paths", type=Path, nargs="+", help="run folders, folders of run folders, or glob patterns")
    parser.add_argument("--out", type=Path, help="output folder (default: <first path>/postpro/<script>)")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    return parser


def find_runs(paths: list[Path]) -> list[Path]:
    """Run folders (those holding solve/) among `paths`, in order, each listed once."""
    runs: list[Path] = []
    for path in paths:
        for match in [Path(p) for p in sorted(glob.glob(str(path)))] or [path]:
            if (match / "solve").is_dir():
                runs.append(match)
            elif match.is_dir():
                runs.extend(sorted(d for d in match.iterdir() if (d / "solve").is_dir()))
    unique = list(dict.fromkeys(r.resolve() for r in runs))
    if not unique:
        raise SystemExit("no run folders (holding solve/) in " + ", ".join(map(str, paths)))
    return unique


def output_dir(out: Path | None, paths: list[Path], runs: list[Path], script: str) -> Path:
    """`out`, or <first path>/postpro/<script> (the first run folder if the first path is a glob pattern)."""
    folder = out or (paths[0] if paths[0].is_dir() else runs[0]) / "postpro" / script
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def solve_dirs(run: Path) -> list[Path]:
    """The run's finished solve folders (with results.json), skipping kept-aside failed attempts."""
    return sorted(p.parent for p in (run / "solve").glob("*/results.json") if not p.parent.name.endswith(".previous"))


def solve_case(solve_dir: Path) -> dict[str, Any]:
    """The solve's settings, reference values and flow condition (solve_case.json)."""
    return json.loads((solve_dir / "solve_case.json").read_text())


def deflections(run: Path) -> dict[str, float]:
    """The run's control surface deflections (deg) from its case.toml; surfaces left out are at 0."""
    path = run / "case.toml"
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return {name: float(angle) for name, angle in tomllib.load(f).get("deflections_deg", {}).items()}


def load_results(runs: list[Path]) -> pl.DataFrame:
    """One row per solved condition: case, condition, folder, alpha, beta, convergence, the six coefficients
    (mean over the final window, and its range), each surface's deflection (<surface>_deg) and hinge moment
    (C_h_<surface>, and M_h_<surface> in N m)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        deflected = deflections(run)
        for folder in solve_dirs(run):
            r = SolveResult.model_validate_json((folder / "results.json").read_text())
            surfaces = list(r.hinge_moment_coefficient)
            rows.append({
                "case": run.name, "condition": folder.name, "folder": str(folder),
                "alpha": r.alpha_deg, "beta": r.beta_deg, "iterations": r.iterations,
                "stopped_early": r.stopped_early, "converged": r.converged,
                **{n: r.coefficients[n].mean for n in COEFFICIENTS},
                **{f"{n}_range": r.coefficients[n].range for n in COEFFICIENTS},
                **{f"{s}_deg": deflected.get(s, 0.0) for s in surfaces},
                **{f"C_h_{s}": r.hinge_moment_coefficient[s].mean for s in surfaces},
                **{f"M_h_{s}": r.hinge_moment_n_m[s].mean for s in surfaces},
            })
    if not rows:
        raise SystemExit("no solve/<condition>/results.json in the run folders")
    return pl.from_dicts(rows, infer_schema_length=None).sort("case", "beta", "alpha")


def surfaces_of(results: pl.DataFrame) -> list[str]:
    """Control surface names, from the C_h_<surface> columns."""
    return [c.removeprefix("C_h_") for c in results.columns if c.startswith("C_h_")]


def print_table(table: pl.DataFrame, title: str) -> None:
    # ASCII: the Windows console can't print Polars' default box-drawing characters.
    print(f"\n{title}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_formatting="ASCII_MARKDOWN",
                   tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, tbl_width_chars=250):
        print(table)


# ---------------------------------------------------------------------------------------------------------------
# Wall surfaces (postProcessing/wallSurfaces/<iteration>/walls.vtp)

_VTK_TYPES: dict[str, str] = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
                              "UInt32": "<u4", "UInt64": "<u8", "UInt8": "u1"}


@dataclass(frozen=True)
class WallSurface:
    """The walls as written by OpenFOAM's surfaces function object: one value of each field per face."""
    points: np.ndarray          # (n_points, 3), metres
    triangles: np.ndarray       # (n_triangles, 3) point indices: each face split into a fan of triangles
    triangle_face: np.ndarray   # (n_triangles,) the face each triangle came from
    normals: np.ndarray         # (n_faces, 3) unit normals, pointing out of the body into the flow
    areas: np.ndarray           # (n_faces,) m^2
    fields: dict[str, np.ndarray]  # per face, e.g. p (n_faces,), wallShearStress (n_faces, 3)


def wall_surface_path(solve_dir: Path) -> Path | None:
    """The solve's walls.vtp at its last written iteration, if it was downloaded."""
    files = sorted(solve_dir.glob("postProcessing/wallSurfaces/*/walls.vtp"), key=lambda p: float(p.parent.name))
    return files[-1] if files else None


def read_vtp(path: Path) -> WallSurface:
    """Read an uncompressed VTK XML PolyData file (inline binary or ASCII) of polygons with cell data."""
    data = path.read_bytes()
    if b"compressor=" in data[:500]:
        raise ValueError(f"{path}: compressed VTK files are not supported")
    header = np.dtype("<u8") if b"header_type='UInt64'" in data[:500] or b'header_type="UInt64"' in data[:500] \
        else np.dtype("<u4")

    arrays: dict[str, np.ndarray] = {}
    for m in re.finditer(rb"<DataArray([^>]*)>(.*?)</DataArray>", data, re.DOTALL):
        attrs = dict(re.findall(r"""(\w+)=['"]([^'"]*)['"]""", m.group(1).decode()))
        dtype = np.dtype(_VTK_TYPES[attrs["type"]])
        if attrs.get("format") == "ascii":
            values = np.array(m.group(2).split(), dtype=dtype)
        else:
            raw = base64.b64decode(b"".join(m.group(2).split()))
            n_bytes = int(np.frombuffer(raw[:header.itemsize], header)[0])
            values = np.frombuffer(raw[header.itemsize:header.itemsize + n_bytes], dtype)
        components = int(attrs.get("NumberOfComponents", "1"))
        arrays[attrs["Name"]] = values.reshape(-1, components) if components > 1 else values

    points = arrays.pop("Points").astype(np.float64)
    connectivity = arrays.pop("connectivity").astype(np.int64)
    ends = arrays.pop("offsets").astype(np.int64)
    arrays.pop("TimeValue", None)
    starts = np.concatenate([[0], ends[:-1]])
    counts = ends - starts

    # Fan triangulation: face f with corners c0..ck gives triangles (c0, ci, ci+1).
    tris: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for i in range(1, int(counts.max()) - 1):
        f = np.nonzero(counts > i + 1)[0]
        tris.append(np.stack([connectivity[starts[f]], connectivity[starts[f] + i],
                              connectivity[starts[f] + i + 1]], axis=1))
        faces.append(f)
    triangles, triangle_face = np.concatenate(tris), np.concatenate(faces)

    corners = points[triangles]
    area_vectors = np.zeros((len(counts), 3))
    np.add.at(area_vectors, triangle_face, np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) / 2)
    areas = np.linalg.norm(area_vectors, axis=1)
    # OpenFOAM's boundary faces point out of the fluid, i.e. into the body; flip them to point into the flow.
    normals = -area_vectors / np.maximum(areas, 1e-300)[:, None]
    fields = {name: values.astype(np.float64) for name, values in arrays.items()}
    return WallSurface(points, triangles, triangle_face, normals, areas, fields)


def cut(surface: WallSurface, y: float) -> pl.DataFrame:
    """The wall surface cut by the plane at spanwise position `y` (m): one row per piece of face it crosses,
    with the piece's midpoint (x, z), its face, and its side ("upper" or "lower"; side faces are dropped)."""
    corners = surface.points[surface.triangles]               # (m, 3, 3)
    d = corners[:, :, 1] - y
    d = np.where(d == 0, 1e-12, d)                            # a corner exactly on the plane counts as above it
    crossing = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    c, d, face = corners[crossing], d[crossing], surface.triangle_face[crossing]

    # Each crossing triangle has exactly two edges that cross the plane; their crossing points end the piece.
    ends = []
    valid = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        with np.errstate(divide="ignore", invalid="ignore"):  # edges that don't cross are discarded below
            t = d[:, a] / (d[:, a] - d[:, b])
        ends.append(c[:, a] + t[:, None] * (c[:, b] - c[:, a]))
        valid.append(d[:, a] * d[:, b] < 0)
    ends_arr, valid_arr = np.stack(ends, axis=1), np.stack(valid, axis=1)
    first_two = np.argsort(~valid_arr, axis=1, kind="stable")[:, :2]
    mid = np.take_along_axis(ends_arr, first_two[:, :, None], axis=1).mean(axis=1)

    nz = surface.normals[face, 2]
    side = np.where(nz > SURFACE_NORMAL_MIN, "upper", np.where(nz < -SURFACE_NORMAL_MIN, "lower", ""))
    keep = side != ""
    return pl.DataFrame({"x": mid[keep, 0], "z": mid[keep, 2], "face": face[keep], "side": side[keep]})


def chordwise(surface: WallSurface, y: float, values: np.ndarray) -> pl.DataFrame:
    """A per-face quantity along the cut at `y` (m): x (m), x/c from the cut's leading edge (0) to its trailing
    edge (1), side, and value, sorted by x."""
    pieces = cut(surface, y)
    if pieces.is_empty():
        return pl.DataFrame(schema={"x": pl.Float64, "x_c": pl.Float64, "side": pl.String, "value": pl.Float64})
    x_le, x_te = pieces["x"].min(), pieces["x"].max()
    return (pieces.with_columns(value=pl.Series(values[pieces["face"].to_numpy()]),
                                x_c=(pl.col("x") - x_le) / (x_te - x_le))
            .select("x", "x_c", "side", "value").sort("x"))


def common_conditions(runs: list[Path], wanted: list[str] | None) -> list[str]:
    """The flow conditions to compare: `wanted`, or every condition solved in all the runs."""
    solved = [{d.name for d in solve_dirs(r)} for r in runs]
    if wanted:
        return wanted
    common = sorted(set.intersection(*solved))
    if not common:
        raise SystemExit("the runs share no solved condition; choose some with --conditions")
    return common
