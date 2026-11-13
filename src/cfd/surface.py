"""Indexed-mesh helpers for triangulated surfaces: welding corners into vertices, and listing edges.

An STL repeats corner coordinates in every triangle that uses them, so corners with the same coordinates are
first "welded" into shared, numbered vertices. Each triangle then becomes three vertex numbers, and its three
sides become edges between vertex numbers. On a closed surface every edge is shared by an even number of
triangles (two, or four where two parts of the surface touch along a line); an edge used once lies on the rim of
a hole.
"""

import torch

from src.cfd.stl import Triangles

# Corners whose coordinates agree to this many decimal places are treated as the same vertex. The STL values
# are float32, so genuinely shared corners are bit-identical and any small value works.
WELD_DECIMALS = 9


def weld(triangles: Triangles) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert to an indexed mesh: `vertices` (m, 3) unique points and `faces` (n, 3) vertex numbers."""
    corners = triangles.reshape(-1, 3).round(decimals=WELD_DECIMALS)
    vertices, vertex_of_corner = torch.unique(corners, dim=0, return_inverse=True)
    return vertices, vertex_of_corner.view(-1, 3)


def directed_edges(faces: torch.Tensor) -> torch.Tensor:
    """Every triangle side as a (from, to) vertex pair, in the triangle's own corner order: shape (3n, 2)."""
    return torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def open_edge_count(triangles: Triangles) -> int:
    """Number of edges on the rim of a hole (used by an odd number of triangles). 0 means the surface is closed."""
    _, faces = weld(triangles)
    undirected = directed_edges(faces).sort(dim=1).values  # (a, b) and (b, a) become the same edge
    _, uses = torch.unique(undirected, dim=0, return_counts=True)
    return int((uses % 2 == 1).sum())
