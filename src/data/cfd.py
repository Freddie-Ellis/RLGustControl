"""Extract OpenFOAM CFD batches into Parquet.

A batch is one mesh (one set of control-surface deflections) solved at several flow conditions:

    <batch>/
        case.toml                    [deflections_deg] surface = deg, a surface left out is at 0 deg
        mesh/constant/polyMesh/      boundary, points, faces: which wall patch each sampled face is on
        solve/<run>/results.json     alpha, beta, speed, convergence and time-averaged loads
        solve/<run>/postProcessing/wallSurfaces/<iteration>/walls.vtp   p on every wall face

Output, in `out_dir`:
    runs.parquet                one row per solve: settings, deflections, coefficients, forces, moments, hinge moments
    surface/<run>.parquet       one row per wall face: patch, centroid, unit normal, area, p and Cp

walls.vtp merges every wall patch (in the order the parallel sampling gathered them), so each face's patch is found
by matching its centroid to the wall faces of the batch's mesh, which is read once per batch. Apart from that mesh,
only these small files are read; the field directories (U, p, ... per iteration) are never opened, so a batch on a
cloud-synced share only downloads what is used. Runs already extracted are skipped, so new batches can be added to
the same `out_dir` incrementally.
"""

import base64
import json
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT: Path = Path(__file__).resolve().parent.parent
CFD_SOURCE_DIR: Path = Path(r"C:\dev\mesh_coarse")  # a batch, or a folder of batches
CFD_DATA_DIR: Path = ROOT / "data" / "cfd"

RUNS_FILE: str = "runs.parquet"
SURFACE_DIR: str = "surface"
MESH_DIR: Path = Path("mesh") / "constant" / "polyMesh"  # inside a batch

_VTK_DTYPES: dict[str, str] = {
    "Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt32": "<u4", "UInt64": "<u8",
}
_HEADER_DTYPES: dict[str, str] = {"UInt32": "<u4", "UInt64": "<u8"}


# ---- VTK PolyData (.vtp) --------------------------------------------------------------------------------------------

def _decode(array: ET.Element, header_type: str) -> np.ndarray:
    """One inline binary DataArray: base64 of [byte count header][raw little-endian data]."""
    if array.get("format") != "binary":
        raise ValueError(f"DataArray {array.get('Name')!r}: only format='binary' is supported")
    raw = base64.b64decode("".join((array.text or "").split()))
    header = np.dtype(_HEADER_DTYPES[header_type])
    n_bytes = int(np.frombuffer(raw[: header.itemsize], dtype=header)[0])
    data = np.frombuffer(raw[header.itemsize : header.itemsize + n_bytes], dtype=_VTK_DTYPES[array.get("type", "")])
    n_components = int(array.get("NumberOfComponents", "1"))
    return data.reshape(-1, n_components) if n_components > 1 else data


def read_vtp_faces(path: Path) -> dict[str, np.ndarray]:
    """Faces of an uncompressed .vtp surface: centroid (n, 3), area_normal (n, 3, area-weighted) and every CellData
    array by name."""
    root = ET.parse(path).getroot()
    if root.get("compressor"):
        raise ValueError(f"{path}: compressed .vtp is not supported")
    header_type = root.get("header_type", "UInt32")
    piece = root.find("PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path}: no PolyData piece")

    def arrays(section: str) -> dict[str, np.ndarray]:
        return {a.get("Name", ""): _decode(a, header_type) for a in piece.findall(f"{section}/DataArray")}

    points = arrays("Points")["Points"].astype(np.float64)
    polys = arrays("Polys")
    connectivity, ends = polys["connectivity"], polys["offsets"]
    starts = np.concatenate([[0], ends[:-1]])
    corners = points[connectivity]
    # Newell's method: half the sum of cross products of consecutive corners is the area-weighted normal.
    following = np.arange(1, len(connectivity) + 1)
    following[ends - 1] = starts
    area_normal = 0.5 * np.add.reduceat(np.cross(corners, corners[following]), starts)
    return {"centroid": _face_centroids(points, connectivity, ends), "area_normal": area_normal} | arrays("CellData")


def _face_centroids(points: np.ndarray, connectivity: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean corner of each face (n, 3). `ends[i]` is one past face i's last entry in `connectivity`.

    The .vtp and the mesh both go through this from the same float32 points, so a face gets bit-identical centroids
    from either and they can be matched exactly."""
    starts = np.concatenate([[0], ends[:-1]])
    return np.add.reduceat(points[connectivity], starts) / np.diff(np.concatenate([[0], ends]))[:, None]


# ---- OpenFOAM mesh: the wall patch of each face ---------------------------------------------------------------------

_FOAM_LIST = re.compile(rb"\s*(?://[^\n]*\n\s*)*(\d+)\s*\(")  # a count then '(', after any // comment lines


def _read_foam_lists(path: Path, width: int = 1) -> list[np.ndarray]:
    """Every list in a binary OpenFOAM file, in order. A vectorField (`width` 3) is one (n, 3) list of scalars; a
    faceCompactList is two lists of labels: offsets (n_faces + 1), then the vertex labels of every face."""
    raw = path.read_bytes()
    body = raw.index(b"}", raw.index(b"FoamFile")) + 1
    header = raw[:body].decode(errors="replace")
    if re.search(r"format\s+binary;", header) is None:
        raise ValueError(f"{path}: only format binary is supported")
    bits = dict(re.findall(r"(label|scalar)=(\d+)", header))
    is_labels = re.search(r"class\s+faceCompactList;", header) is not None
    dtype = np.dtype(f"<i{int(bits['label']) // 8}" if is_labels else f"<f{int(bits['scalar']) // 8}")
    lists: list[np.ndarray] = []
    pos = body
    while match := _FOAM_LIST.match(raw, pos):
        n = int(match.group(1)) * width
        lists.append(np.frombuffer(raw, dtype=dtype, count=n, offset=match.end()))
        pos = match.end() + n * dtype.itemsize + 1  # past the closing ')'
    return [a.reshape(-1, width) if width > 1 else a for a in lists]


def wall_faces(mesh_dir: Path) -> pl.DataFrame:
    """Centroid (x, y, z) and patch of every face on a wall patch of a reconstructed mesh."""
    boundary = (mesh_dir / "boundary").read_text()
    patches = re.findall(r"(\w+)\s*\{\s*type\s+wall;[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", boundary)
    # the .vtp stores points as float32; round them the same way so the centroids match bit for bit
    points = _read_foam_lists(mesh_dir / "points", width=3)[0].astype(np.float32).astype(np.float64)
    offsets, labels = _read_foam_lists(mesh_dir / "faces")
    frames: list[pl.DataFrame] = []
    for name, n_faces, start_face in patches:
        ends = offsets[int(start_face) : int(start_face) + int(n_faces) + 1]
        centroid = _face_centroids(points, labels[ends[0] : ends[-1]], ends[1:] - ends[0])
        frames.append(pl.DataFrame({"x": centroid[:, 0], "y": centroid[:, 1], "z": centroid[:, 2], "patch": name}))
    return pl.concat(frames)


def face_patches(centroid: np.ndarray, walls: pl.DataFrame) -> pl.Series:
    """Patch of each sampled face (n,), from its centroid (n, 3) and the mesh's `wall_faces`.

    A mesh can hold coincident wall faces (same centroid, e.g. where two bodies touch). They are matched once, to the
    first patch listed, so the result has one row per sampled face; coincident faces on different patches are
    reported, since their patch is then ambiguous."""
    centroid_cols = ["x", "y", "z"]
    shared = walls.filter(pl.struct(centroid_cols).is_duplicated())
    if clashes := shared.group_by(centroid_cols).agg(pl.col("patch").unique()).filter(pl.col("patch").list.len() > 1).height:
        print(f"warning: {clashes} wall face centroid(s) are shared by different patches; using the first patch listed")
    walls = walls.unique(subset=centroid_cols, keep="first", maintain_order=True)
    faces = pl.DataFrame({"x": centroid[:, 0], "y": centroid[:, 1], "z": centroid[:, 2]})
    patch = faces.join(walls, on=centroid_cols, how="left", maintain_order="left")["patch"]
    if missing := patch.null_count():
        raise ValueError(f"{missing} of {len(patch)} sampled faces match no wall face of the mesh")
    return patch


# ---- One solve ------------------------------------------------------------------------------------------------------

def _final_surface(run_dir: Path) -> Path:
    """walls.vtp of the last iteration written."""
    surfaces = run_dir / "postProcessing" / "wallSurfaces"
    last = max((d for d in surfaces.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    return last / "walls.vtp"


def run_row(run: str, batch_dir: Path, run_dir: Path, deflections: dict[str, float]) -> dict[str, Any]:
    """Settings and time-averaged loads of one solve, flattened to one row."""
    res = json.loads((run_dir / "results.json").read_text())
    surfaces = list(res["hinge_moment_n_m"])  # the control surfaces, in the solver's order
    if unknown := set(deflections) - set(surfaces):
        raise ValueError(f"{batch_dir / 'case.toml'}: unknown surfaces {sorted(unknown)}; expected {surfaces}")
    row: dict[str, Any] = {
        "run": run,
        "batch": batch_dir.name,
        "solve": run_dir.name,
        "alpha_deg": res["alpha_deg"],
        "beta_deg": res["beta_deg"],
        "speed_m_s": res["speed_m_s"],
        "iterations": res["iterations"],
        "converged": res["converged"],
    }
    row |= {f"deflection_{s}_deg": float(deflections.get(s, 0.0)) for s in surfaces}
    row |= {name: c["mean"] for name, c in res["coefficients"].items()}
    row |= {f"force_{axis}_n": v["mean"] for axis, v in res["force_n"].items()}
    row |= {f"moment_{axis}_n_m": v["mean"] for axis, v in res["moment_n_m"].items()}
    row |= {f"hinge_moment_{s}_n_m": v["mean"] for s, v in res["hinge_moment_n_m"].items()}
    row |= {f"C_h_{s}": v["mean"] for s, v in res["hinge_moment_coefficient"].items()}
    return row


def surface_table(vtp: Path, speed_m_s: float, walls: pl.DataFrame) -> pl.DataFrame:
    """Pressure on every wall face, with the patch it is on (from the batch's `wall_faces`). p is kinematic (p / rho) relative to the far field, so Cp = p / (U^2 / 2).

    The normal points out of the body into the flow, so the pressure force is -rho * sum(p * n * area).
    """
    faces = read_vtp_faces(vtp)
    area = np.linalg.norm(faces["area_normal"], axis=1)
    normal = -faces["area_normal"] / area[:, None]  # OpenFOAM wall normals point out of the fluid, into the body
    p = faces["p"].astype(np.float32)
    return pl.DataFrame({
        "face": np.arange(len(p), dtype=np.int32),
        "patch": face_patches(faces["centroid"], walls),
        "x": faces["centroid"][:, 0].astype(np.float32),
        "y": faces["centroid"][:, 1].astype(np.float32),
        "z": faces["centroid"][:, 2].astype(np.float32),
        "nx": normal[:, 0].astype(np.float32),
        "ny": normal[:, 1].astype(np.float32),
        "nz": normal[:, 2].astype(np.float32),
        "area": area.astype(np.float32),
        "p": p,
        "cp": p / np.float32(0.5 * speed_m_s**2),
    })


# ---- Batches --------------------------------------------------------------------------------------------------------

def batch_dirs(source: Path) -> list[Path]:
    """`source` itself if it is a batch, otherwise every batch directly inside it."""
    if (source / "solve").is_dir():
        return [source]
    return sorted(d for d in source.iterdir() if (d / "solve").is_dir())


def surface_file(run: str, out_dir: Path = CFD_DATA_DIR) -> Path:
    return out_dir / SURFACE_DIR / f"{run}.parquet"


def extract(source: Path = CFD_SOURCE_DIR, out_dir: Path = CFD_DATA_DIR) -> pl.DataFrame:
    """Extract every solve under `source` into `out_dir`, skipping runs already there. Returns the full run table."""
    surface_dir = out_dir / SURFACE_DIR
    surface_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / RUNS_FILE
    runs = pl.read_parquet(runs_path) if runs_path.exists() else pl.DataFrame()
    done = set(runs["run"]) if runs.height else set()

    new_rows: list[dict[str, Any]] = []
    for batch in batch_dirs(source):
        case = tomllib.loads((batch / "case.toml").read_text())
        deflections: dict[str, float] = case.get("deflections_deg", {})
        walls: pl.DataFrame | None = None  # read on the batch's first new run, shared by all its runs
        for run_dir in sorted(d for d in (batch / "solve").iterdir() if d.is_dir()):
            run = f"{batch.name}__{run_dir.name}"
            if run in done:
                continue
            if not (run_dir / "results.json").exists():
                print(f"skip {run}: no results.json (solve failed or still running)")
                continue
            row = run_row(run, batch, run_dir, deflections)
            if walls is None:
                walls = wall_faces(batch / MESH_DIR)
            surface_table(_final_surface(run_dir), row["speed_m_s"], walls).write_parquet(surface_file(run, out_dir))
            new_rows.append(row)
            print(f"extracted {run}{'' if row['converged'] else '  (NOT converged)'}")

    if new_rows:
        runs = pl.concat([runs, pl.DataFrame(new_rows)], how="diagonal_relaxed")
        runs.write_parquet(runs_path)
    return runs


def load_surface(out_dir: Path = CFD_DATA_DIR) -> pl.LazyFrame:
    """Every run's surface table as one lazy frame, with a `run` column to join on runs.parquet."""
    return pl.scan_parquet(out_dir / SURFACE_DIR / "*.parquet", include_file_paths="path").with_columns(
        run=pl.col("path").str.extract(r"([^/\\]+)\.parquet$")
    ).drop("path")


def main() -> None:
    runs = extract()
    print(runs.select("run", "alpha_deg", "beta_deg", "converged", "C_L", "C_m"))


if __name__ == "__main__":
    main()
