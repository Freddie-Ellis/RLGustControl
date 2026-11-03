"""Reduce the touching bodies of the aircraft to its outer (wetted) skin before meshing.

The aircraft is modelled as separate closed bodies that touch: a flap's round nose sits flush in its cove,
its end faces sit flush against the main body, and when deflected part of it is buried inside the main body.
snappyHexMesh would see those hidden faces too. Two faces lying on top of each other with opposite normals
look like extremely sharp curvature, so it refines around them to the maximum level, and the refinement
spills out onto the outer skin as dense bands along every seam, even at zero deflection.

`outer_skin` removes, from every body, the faces that are hidden:
    flush   lying against another body's face (facing it, closer than flush_tol);
    buried  inside another body.
What remains is the wetted surface, still split by body, so each body keeps its own patch. The skin is no
longer closed: it has open edges along the seams between bodies, and cracks there of the order of the STL
export deviation (well below any cell size), which snappyHexMesh does not mind.
"""

import torch

from src.cfd.geometry_check import distance_to_facing
from src.cfd.stl import Triangles, facet_normals

# Ray directions for the inside test: tilted slightly off the axes so rays don't run along mesh edges.
# A point counts as inside if most rays cross the surface an odd number of times.
_RAYS = torch.tensor([[0.0123, 0.0217, 1.0], [0.0191, -0.0137, -1.0], [1.0, 0.0173, 0.0229]], dtype=torch.float64)
_RAYS = _RAYS / _RAYS.norm(dim=1, keepdim=True)


def _crossings(points: torch.Tensor, triangles: Triangles, direction: torch.Tensor) -> torch.Tensor:
    """Number of times a ray from each point along `direction` crosses the triangles (Moller-Trumbore)."""
    a, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    pvec = torch.linalg.cross(direction.expand_as(e2), e2)
    det = (e1 * pvec).sum(-1)
    ok = det.abs() > 1e-30
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    tvec = points[:, None, :] - a[None]
    u = (tvec * pvec[None]).sum(-1) * inv
    qvec = torch.linalg.cross(tvec, e1[None].expand_as(tvec))
    v = (qvec * direction).sum(-1) * inv
    t = (qvec * e2[None]).sum(-1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return hit.sum(dim=1)


def inside(points: torch.Tensor, closed: Triangles, chunk: int = 256, tri_chunk: int = 8192) -> torch.Tensor:
    """Whether each point lies inside the closed surface `closed` (majority vote over three rays)."""
    votes = torch.zeros(len(points), dtype=torch.int64)
    corners = closed.reshape(-1, 3, 3)
    lo, hi = corners.min(dim=1).values, corners.max(dim=1).values
    span = float((closed.reshape(-1, 3).max(dim=0).values - closed.reshape(-1, 3).min(dim=0).values).norm())
    for direction in _RAYS:
        # Only triangles near the ray's line can be crossed: keep those whose bounding box is within the
        # ray's sideways drift of the points' box, measured across the ray (in the two other axes).
        axis = int(direction.abs().argmax())
        others = [i for i in range(3) if i != axis]
        drift = span * float(direction[others].abs().max()) + 1e-9
        for i in range(0, len(points), chunk):
            p = points[i:i + chunk]
            p_lo, p_hi = p.min(dim=0).values, p.max(dim=0).values
            near = torch.ones(len(closed), dtype=torch.bool)
            for k in others:
                near &= (hi[:, k] >= p_lo[k] - drift) & (lo[:, k] <= p_hi[k] + drift)
            candidates = closed[near]
            crossings = torch.zeros(len(p), dtype=torch.int64)
            for j in range(0, len(candidates), tri_chunk):
                crossings += _crossings(p, candidates[j:j + tri_chunk], direction)
            votes[i:i + chunk] += crossings % 2
    return votes >= 2


def hidden_faces(this: Triangles, other: Triangles, flush_tol: float) -> torch.Tensor:
    """Mask of `this` body's faces that are flush against, or buried inside, the closed body `other`."""
    hidden = torch.zeros(len(this), dtype=torch.bool)
    margin = 10 * flush_tol
    corners = other.reshape(-1, 3)
    lo, hi = corners.min(dim=0).values - margin, corners.max(dim=0).values + margin
    centroids = this.mean(dim=1)
    candidates = ((centroids >= lo) & (centroids <= hi)).all(dim=1).nonzero().flatten()
    if len(candidates) == 0:
        return hidden

    c, n = centroids[candidates], facet_normals(this[candidates])
    # Flush: facing a face of `other` closer than flush_tol (only `other`'s faces near the candidates matter).
    c_lo, c_hi = c.min(dim=0).values - margin, c.max(dim=0).values + margin
    other_centroids = other.mean(dim=1)
    other_near = other[((other_centroids >= c_lo) & (other_centroids <= c_hi)).all(dim=1)]
    if len(other_near):
        flush = distance_to_facing(c, n, other_near, across_only=False) < flush_tol
    else:
        flush = torch.zeros(len(candidates), dtype=torch.bool)
    # Buried: inside `other`. Test the rest only, since flush faces sit on its surface where the test is unclear.
    buried = torch.zeros(len(candidates), dtype=torch.bool)
    rest = ~flush
    if rest.any():
        buried[rest] = inside(c[rest], other)
    hidden[candidates] = flush | buried
    return hidden


def outer_skin(bodies: dict[str, Triangles], flush_tol: float) -> dict[str, Triangles]:
    """Each body's exposed faces only. `bodies` must each be closed; `flush_tol` is in the same units.

    Every body is tested against every other closed body as originally given, so faces two bodies share
    are removed from both.
    """
    skin: dict[str, Triangles] = {}
    for name, triangles in bodies.items():
        hidden = torch.zeros(len(triangles), dtype=torch.bool)
        for other_name, other in bodies.items():
            if other_name != name:
                hidden |= hidden_faces(triangles, other, flush_tol)
        skin[name] = triangles[~hidden]
    return skin
