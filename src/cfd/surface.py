"""Checking that a triangulated surface is closed, and filling small holes so that it is.

A body's surface must be closed (watertight) to mesh it reliably. SolidWorks sometimes leaves thin slits where
two faces meet at a shallow angle, such as where a loft blends into the wing, and these are filled here.

How the check works
-------------------
An STL repeats corner coordinates in every triangle that uses them, so we first "weld" corners with the same
coordinates into shared, numbered vertices. Each triangle then becomes three vertex numbers, and its three
sides become edges between vertex numbers.

On a closed surface every edge is shared by exactly two triangles. An edge used by only one triangle lies on
the rim of a hole (a "boundary edge"). An edge used by three or more is non-manifold, which can't be fixed
automatically.
"""

import torch

from src.cfd.stl import Triangles

# Corners whose coordinates agree to this many decimal places are treated as the same vertex. The STL values
# are float32, so genuinely shared corners are bit-identical and any small value works.
WELD_DECIMALS = 9


def _weld(triangles: Triangles) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert to an indexed mesh: `vertices` (m, 3) unique points and `faces` (n, 3) vertex numbers."""
    corners = triangles.reshape(-1, 3).round(decimals=WELD_DECIMALS)
    vertices, vertex_of_corner = torch.unique(corners, dim=0, return_inverse=True)
    return vertices, vertex_of_corner.view(-1, 3)


def _directed_edges(faces: torch.Tensor) -> torch.Tensor:
    """Every triangle side as a (from, to) vertex pair, in the triangle's own corner order: shape (3n, 2).

    On a consistently oriented closed surface, two triangles sharing an edge traverse it in opposite directions.
    """
    return torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def open_edge_count(triangles: Triangles) -> int:
    """Number of edges not shared by exactly two triangles. 0 means the surface is closed."""
    _, faces = _weld(triangles)
    undirected = _directed_edges(faces).sort(dim=1).values  # (a, b) and (b, a) become the same edge
    _, uses = torch.unique(undirected, dim=0, return_counts=True)
    return int((uses != 2).sum())


def close_surface(triangles: Triangles, max_hole_perimeter: float) -> Triangles:
    """Return the surface with every hole filled, or raise if a hole is larger than `max_hole_perimeter`.

    Holes are filled with new triangles appended to the end. A closed surface is returned unchanged. The limit
    stops a genuine opening (a missing face, say) from being silently covered over; set it to 0 to require
    every surface to be closed already.
    """
    vertices, faces = _weld(triangles)
    edges = _directed_edges(faces)

    # Count how many triangles use each edge, ignoring direction.
    _, edge_number, uses = torch.unique(edges.sort(dim=1).values, dim=0, return_inverse=True, return_counts=True)
    if (uses > 2).any():
        raise ValueError(f"{int((uses > 2).sum())} non-manifold edges (shared by more than two triangles)")
    boundary_edges = edges[uses[edge_number] == 1]
    if len(boundary_edges) == 0:
        return triangles

    patches: list[tuple[int, int, int]] = []
    for loop in _trace_hole_loops(boundary_edges.tolist()):
        points = vertices[loop]
        perimeter = float((points - points.roll(-1, dims=0)).norm(dim=1).sum())
        if perimeter > max_hole_perimeter:
            centre = [round(c, 2) for c in points.mean(dim=0).tolist()]
            raise ValueError(
                f"hole with perimeter {perimeter:.3g} near {centre} exceeds max_hole_perimeter {max_hole_perimeter}"
            )
        patches.extend(_fill_loop(loop))
    return torch.cat([triangles, vertices[torch.tensor(patches)]])


def _trace_hole_loops(boundary_edges: list[list[int]]) -> list[list[int]]:
    """Chain boundary edges into closed loops of vertex numbers, one loop per hole.

    Each loop runs in the opposite direction to the existing triangles' edges around the hole. A patch
    triangle that follows the loop's order therefore faces the same way (outward) as its neighbours.

    Two slits can touch at a single vertex (a pinch point), so a vertex may have more than one way onwards.
    Whenever the walk comes back to a vertex already on the current path, the section in between is cut off
    as its own loop, which keeps every loop simple (no repeated vertices).
    """
    # next_vertices[v] lists the vertices the walk can step to from v: the reverse of each boundary edge into v.
    next_vertices: dict[int, list[int]] = {}
    for start, end in boundary_edges:
        next_vertices.setdefault(end, []).append(start)

    loops: list[list[int]] = []
    while next_vertices:
        first = next(iter(next_vertices))
        path = [first]
        index_on_path = {first: 0}
        while True:
            here = path[-1]
            if here not in next_vertices:
                raise ValueError("hole rims do not form closed loops (inconsistent triangle orientation?)")
            step = next_vertices[here].pop()  # use up this boundary edge
            if not next_vertices[here]:
                del next_vertices[here]

            if step not in index_on_path:
                index_on_path[step] = len(path)
                path.append(step)
                continue

            # Back at a vertex already on the path: everything from there onwards closes a loop.
            k = index_on_path[step]
            loops.append(path[k:])
            for vertex in path[k + 1:]:
                del index_on_path[vertex]
            del path[k + 1:]
            if k == 0 and first not in next_vertices:
                break  # back at the start with nothing left to follow from here
    return loops


def _fill_loop(loop: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate a hole by zig-zagging inwards from both ends of the loop.

    The holes here are long thin slits, and a zig-zag strip across them gives triangles that span the slit's
    width. A fan from one corner would give long needle-shaped triangles. For loop vertices 0..5:

        0 - 1 - 2           triangles (0,1,5), (1,4,5), (1,2,4), (2,3,4)
        |       |
        5 - 4 - 3

    Each triangle keeps the loop's direction, so it is oriented consistently with its neighbours.
    """
    triangles: list[tuple[int, int, int]] = []
    i, j = 0, len(loop) - 1
    while j - i >= 2:
        triangles.append((loop[i], loop[i + 1], loop[j]))
        i += 1
        if j - i >= 2:
            triangles.append((loop[i], loop[j - 1], loop[j]))
            j -= 1
    return triangles
