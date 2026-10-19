"""Reading and writing STL files.

An STL file is just a list of triangles (facets), each given by its three corner points and a normal. There
is no shared vertex list, so neighbouring triangles repeat the same corner coordinates.

Throughout the CFD code a surface is held as a float64 tensor of shape (n, 3, 3):
    triangles[i]        -> the i-th triangle
    triangles[i, j]     -> its j-th corner, j = 0, 1, 2
    triangles[i, j, :]  -> that corner's (x, y, z)
Corners are ordered anticlockwise when seen from outside (right-hand rule), so the normal points outward.

Two file formats exist:
    ASCII:  readable text made of named solids ("solid main_body ... endsolid main_body"), each a list of
            "facet normal / vertex x y z" lines. One file can hold several named solids.
    Binary: an 80-byte header, a uint32 triangle count, then one 50-byte record per triangle:
            normal (3 x float32), three corners (9 x float32), a 2-byte attribute (unused). No solid names.

SolidWorks exports are read in either format. The pipeline writes ASCII, because it puts every body in one
file as its own named solid, and OpenFOAM turns each solid into a separate patch.
"""

import re
import struct
from pathlib import Path

import torch

type Triangles = torch.Tensor  # (n, 3, 3) float64, see module docstring

HEADER_BYTES = 80
COUNT_BYTES = 4                 # uint32 number of triangles, straight after the header
RECORD_BYTES = 50               # bytes per triangle in a binary STL
CORNER_BYTES = slice(12, 48)    # where the 9 corner floats sit in a record (after the 3 normal floats)

# Matches "vertex <x> <y> <z>" lines in an ASCII STL; the stored normals are ignored and recomputed.
_ASCII_VERTEX = re.compile(rb"^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)


def read_stl(path: Path) -> Triangles:
    """Read an ASCII or binary STL file into an (n, 3, 3) float64 tensor (all solids together)."""
    data = path.read_bytes()

    # A binary file's size is fully determined by its triangle count. We can't detect the format from the
    # first bytes, because SolidWorks writes binary files whose header also starts with "solid".
    if len(data) >= HEADER_BYTES + COUNT_BYTES:
        (n,) = struct.unpack_from("<I", data, HEADER_BYTES)
        if len(data) == HEADER_BYTES + COUNT_BYTES + n * RECORD_BYTES:
            records = torch.frombuffer(bytearray(data[HEADER_BYTES + COUNT_BYTES:]), dtype=torch.uint8)
            corners = records.view(n, RECORD_BYTES)[:, CORNER_BYTES].contiguous().view(torch.float32)
            return corners.view(n, 3, 3).to(torch.float64)

    # Otherwise treat it as ASCII: every 3 "vertex" lines form one triangle.
    coords = [float(c) for match in _ASCII_VERTEX.finditer(data) for c in match.groups()]
    if not coords or len(coords) % 9 != 0:
        raise ValueError(f"{path}: not a valid STL (found {len(coords) // 3} vertices)")
    return torch.tensor(coords, dtype=torch.float64).view(-1, 3, 3)


def facet_normals(triangles: Triangles) -> torch.Tensor:
    """Outward unit normal of each triangle, shape (n, 3), from its corner order. Zero-area triangles get 0."""
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    normal = torch.linalg.cross(edge_1, edge_2)
    return normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-300)


def write_ascii_stl(path: Path, solids: dict[str, Triangles]) -> None:
    """Write an ASCII STL with one named solid per entry of `solids` ({name: triangles}).

    Coordinates are written to 8 significant figures, beyond the float32 precision of the SolidWorks exports.
    """
    lines: list[str] = []
    for name, triangles in solids.items():
        lines.append(f"solid {name}")
        for (nx, ny, nz), corners in zip(facet_normals(triangles).tolist(), triangles.tolist()):
            lines.append(f"  facet normal {nx:.7e} {ny:.7e} {nz:.7e}")
            lines.append("    outer loop")
            lines.extend(f"      vertex {x:.7e} {y:.7e} {z:.7e}" for x, y, z in corners)
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
