"""Pre-meshing checks of a model's STLs: every body is closed, and no control surface leaves a gap that is too
thin to mesh at any deflection.

Why gaps matter: where a control surface faces another body across a gap about as wide as the mesh cells
(1-2 mm), snappyHexMesh over-refines the gap and still can't fill it with good cells, which ruins the mesh
around it. Surfaces that touch (flush) or are far apart are fine.

For each control surface, at a sweep of angles across its deflection limits (every other surface at 0), this
finds where it faces another body across a parallel-walled gap and measures the gap's width straight across
(along the normal). Walls meeting at a larger angle form a corner or wedge, such as beside a deflected hinge,
which snappyHexMesh handles like any other corner, so they are not counted. Contact is sorted into:
    flush      closer than flush_tol (default 0.3 mm, under a third of the finest 1 mm cell, so the mesh treats
               it as touching). This also covers two triangulations of the same touching surface, which can
               sit up to about twice the STL export deviation apart, and the narrow strip either side of the
               contact line where a rotated round flap nose meets the body.
    thin gap   between flush_tol and max_gap (default 2 mm): about the cell size, so a problem.
    (wider gaps are open space and meshed normally)

Run on a geometry (exits with status 1 if there are problems):
    uv run python -m src.cfd.geometry_check cad/uav/geometry.toml
    uv run python -m src.cfd.geometry_check cad/uav/geometry.toml --angles 9 --write-gaps gaps.stl
"""

import argparse
import sys
from pathlib import Path

import torch

from src.cfd.deflect import deflect
from src.cfd.geometry import Body, GeometryConfig, StrictModel
from src.cfd.stl import Triangles, facet_normals, read_stl, write_ascii_stl
from src.cfd.surface import close_surface, open_edge_count

# Two walls form a gap (rather than a corner) if they are within about 18 degrees of parallel and face each
# other, i.e. their normals are more than 162 degrees apart.
FACING_COS = -0.95


class BodyCheck(StrictModel):
    name: str
    facets: int
    open_edges: int        # in the STL as exported
    patch_facets: int      # triangles added to fill small holes (0 if already closed)
    error: str | None      # why the surface couldn't be closed, if it couldn't


class JoinCheck(StrictModel):
    """Contact between one control surface, at one angle, and one other body."""
    surface: str
    other: str
    angle_deg: float
    flush_area: float              # units^2
    gap_area: float                # units^2 of thin gap
    narrowest_gap: float | None    # smallest thin-gap distance, in units
    gap_location: tuple[float, float, float] | None


class GeometryCheckReport(StrictModel):
    units: str
    flush_tol: float
    max_gap: float
    bodies: list[BodyCheck]
    joins: list[JoinCheck]
    problems: list[str]

    @property
    def passed(self) -> bool:
        return not self.problems


def _areas(triangles: Triangles) -> torch.Tensor:
    return 0.5 * torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]).norm(dim=1)


def _distance_across(points: torch.Tensor, triangles: Triangles) -> torch.Tensor:
    """Distance from each point (p, 3) straight across to each triangle (t, 3, 3), shape (p, t): the distance
    to the point's projection onto the triangle's plane, or infinity if that projection falls outside the
    triangle. A point beside a surface's edge is therefore never counted as being in a gap with it.
    """
    p = points[:, None, :]
    a, b, c = triangles[None, :, 0], triangles[None, :, 1], triangles[None, :, 2]
    ab, ac, ap = b - a, c - a, p - a
    # Barycentric coordinates (u, v, w) of the projection; it lies inside the triangle if all are >= 0.
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-300)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    normal = torch.linalg.cross(ab, ac)
    height = ((ap * normal).sum(-1) / normal.norm(dim=-1).clamp_min(1e-300)).abs()
    return torch.where(inside, height, torch.inf)


def _closest_distance(points: torch.Tensor, triangles: Triangles) -> torch.Tensor:
    """Distance from each point (p, 3) to the closest point of each triangle (t, 3, 3), shape (p, t).

    Uses the barycentric region test from Ericson, Real-Time Collision Detection (5.1.5): if the point projects
    inside the triangle the distance is to that projection, otherwise to the nearest edge.
    """
    p = points[:, None, :]
    a, b, c = triangles[None, :, 0], triangles[None, :, 1], triangles[None, :, 2]
    ab, ac = b - a, c - a
    d1, d2 = (ab * (p - a)).sum(-1), (ac * (p - a)).sum(-1)
    d3, d4 = (ab * (p - b)).sum(-1), (ac * (p - b)).sum(-1)
    d5, d6 = (ab * (p - c)).sum(-1), (ac * (p - c)).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    denom = (va + vb + vc).clamp_min(1e-300)
    projection = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]

    def to_segment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = y - x
        t = (((p - x) * xy).sum(-1) / (xy * xy).sum(-1).clamp_min(1e-300)).clamp(0, 1)
        return (p - (x + t[..., None] * xy)).norm(dim=-1)

    to_edges = torch.minimum(torch.minimum(to_segment(a, b), to_segment(b, c)), to_segment(c, a))
    return torch.where(inside, (p - projection).norm(dim=-1), to_edges)


def distance_to_facing(
    points: torch.Tensor, normals: torch.Tensor, triangles: Triangles, across_only: bool = True, chunk: int = 1024
) -> torch.Tensor:
    """For each point (with its surface normal), the distance to the nearest triangle facing it (near-parallel,
    normals opposed); infinity if there is none.

    With `across_only` (for measuring gaps), only distances straight across count, so a point beside a
    surface's edge is never in a gap with it. Without it (for deciding whether a face is covered), the plain
    closest distance is used: on curved surfaces made of long thin triangles, such as a flap's round nose, a
    point can project just outside every triangle of a surface it is touching.
    """
    measure = _distance_across if across_only else _closest_distance
    tri_normals = facet_normals(triangles)
    best = torch.full((len(points),), float("inf"), dtype=points.dtype)
    for i in range(0, len(points), chunk):
        for j in range(0, len(triangles), chunk):
            d = measure(points[i:i + chunk], triangles[j:j + chunk])
            facing = normals[i:i + chunk] @ tri_normals[j:j + chunk].T < FACING_COS
            d = torch.where(facing, d, torch.inf)
            best[i:i + chunk] = torch.minimum(best[i:i + chunk], d.min(dim=1).values)
    return best


def _near(triangles: Triangles, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Indices of triangles whose centroid lies in the box [lo, hi]."""
    c = triangles.mean(dim=1)
    return ((c >= lo) & (c <= hi)).all(dim=1).nonzero().flatten()


def measure_join(
    surface: Triangles, other: Triangles, flush_tol: float, max_gap: float
) -> tuple[float, float, float | None, tuple[float, float, float] | None, Triangles]:
    """Flush area, thin-gap area, narrowest thin gap and its location, and the thin-gap triangles.

    Measured from both sides (each surface's triangle centroids to the other surface), taking the larger area,
    so a gap is found whichever side is more finely triangulated.
    """
    margin = max_gap + 1.0
    s_box = (surface.reshape(-1, 3).min(dim=0).values - margin, surface.reshape(-1, 3).max(dim=0).values + margin)
    other_near = other[_near(other, *s_box)]
    if len(other_near) == 0:
        return 0.0, 0.0, None, None, surface[:0]
    o_box = (other_near.reshape(-1, 3).min(dim=0).values - margin, other_near.reshape(-1, 3).max(dim=0).values + margin)
    surface_near = surface[_near(surface, *o_box)]

    flush_area, gap_area = 0.0, 0.0
    narrowest: float | None = None
    location: tuple[float, float, float] | None = None
    gap_triangles: list[Triangles] = []
    for this, that in ((surface_near, other_near), (other_near, surface_near)):
        if len(this) == 0 or len(that) == 0:
            continue
        distance = distance_to_facing(this.mean(dim=1), facet_normals(this), that)
        area = _areas(this)
        flush_area = max(flush_area, float(area[distance < flush_tol].sum()))
        thin = (distance >= flush_tol) & (distance < max_gap)
        gap_area = max(gap_area, float(area[thin].sum()))
        if thin.any():
            k = int(torch.where(thin, distance, torch.inf).argmin())
            if narrowest is None or float(distance[k]) < narrowest:
                narrowest = float(distance[k])
                x, y, z = this[k].mean(dim=0).tolist()
                location = (x, y, z)
            gap_triangles.append(this[thin])
    gaps = torch.cat(gap_triangles) if gap_triangles else surface[:0]
    return flush_area, gap_area, narrowest, location, gaps


def check_geometry(
    geometry: GeometryConfig,
    n_angles: int = 5,
    flush_tol: float = 0.3,
    max_gap: float = 2.0,
    area_tol: float = 1.0,
) -> tuple[GeometryCheckReport, dict[str, Triangles]]:
    """Run all checks. Lengths are in the geometry's units. A thin gap counts as a problem once its area exceeds
    `area_tol` (units^2), which ignores isolated triangles. Also returns the thin-gap triangles, by join."""
    problems: list[str] = []
    offset = torch.tensor(geometry.stl_offset, dtype=torch.float64)

    # 1. Every body is closed (after filling holes up to max_hole_perimeter).
    bodies: dict[str, Triangles] = {}
    body_checks: list[BodyCheck] = []
    for body in geometry.bodies:
        raw = read_stl(geometry.stl_path(body)) - offset
        error: str | None = None
        try:
            closed = close_surface(raw, geometry.max_hole_perimeter)
        except ValueError as e:
            closed, error = raw, str(e)
            problems.append(f"{body.name}: {e}")
        bodies[body.name] = closed
        body_checks.append(BodyCheck(
            name=body.name, facets=len(raw), open_edges=open_edge_count(raw),
            patch_facets=len(closed) - len(raw), error=error,
        ))

    # 2. Joins between each control surface and the other bodies, across its deflection range.
    joins: list[JoinCheck] = []
    gap_faces: dict[str, Triangles] = {}
    for surface in geometry.control_surfaces:
        angles = torch.linspace(surface.min_deflection_deg, surface.max_deflection_deg, n_angles).tolist()
        angles = sorted({round(a, 6) for a in [*angles, 0.0]})
        others: list[Body] = [b for b in geometry.bodies if b.name != surface.name]
        for angle in angles:
            moved = deflect(bodies[surface.name], surface.hinge, angle)
            for other in others:
                flush, gap, narrowest, where, faces = measure_join(moved, bodies[other.name], flush_tol, max_gap)
                if flush == 0.0 and gap == 0.0:
                    continue
                joins.append(JoinCheck(
                    surface=surface.name, other=other.name, angle_deg=angle, flush_area=flush,
                    gap_area=gap, narrowest_gap=narrowest, gap_location=where,
                ))
                if gap > area_tol:
                    assert where is not None and narrowest is not None
                    problems.append(
                        f"{surface.name} at {angle:g} deg: {gap:.0f} {geometry.units}^2 of thin gap to {other.name} "
                        f"(narrowest {narrowest:.2f} {geometry.units} near {tuple(round(c, 1) for c in where)})"
                    )
                    gap_faces[f"gap_{surface.name}_{angle:g}deg_{other.name}".replace("-", "m").replace(".", "p")] = faces

    report = GeometryCheckReport(
        units=geometry.units, flush_tol=flush_tol, max_gap=max_gap,
        bodies=body_checks, joins=joins, problems=problems,
    )
    return report, gap_faces


def format_report(report: GeometryCheckReport) -> str:
    u = report.units
    lines = ["Bodies", f"  {'name':22s} {'facets':>8s} {'open edges':>11s} {'hole patches':>13s}"]
    for b in report.bodies:
        status = f"  ERROR: {b.error}" if b.error else ""
        lines.append(f"  {b.name:22s} {b.facets:8,d} {b.open_edges:11d} {b.patch_facets:13d}{status}")

    lines.append("")
    lines.append(f"Control surface joins (other surfaces at 0; flush < {report.flush_tol:g} {u}, "
                 f"thin gap {report.flush_tol:g}-{report.max_gap:g} {u})")
    current = ""
    for j in report.joins:
        key = f"{j.surface} / {j.other}"
        if key != current:
            current = key
            lines.append(f"  {key}")
            lines.append(f"    {'angle':>7s} {'flush ' + u + '^2':>12s} {'thin gap ' + u + '^2':>15s} {'narrowest':>10s}")
        narrowest = f"{j.narrowest_gap:.2f}" if j.narrowest_gap is not None else "-"
        lines.append(f"    {j.angle_deg:7.1f} {j.flush_area:12.0f} {j.gap_area:15.1f} {narrowest:>10s}")

    lines.append("")
    if report.passed:
        lines.append("PASSED: every body is closed and no thin gaps at any deflection checked")
    else:
        lines.append(f"FAILED: {len(report.problems)} problem(s)")
        lines.extend(f"  - {p}" for p in report.problems)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check a geometry's STLs are closed and free of thin gaps.")
    parser.add_argument("geometry", type=Path, help="path to geometry.toml")
    parser.add_argument("--angles", type=int, default=5, help="angles per control surface across its limits")
    parser.add_argument("--flush-tol", type=float, default=0.3, help="contact closer than this is flush (units)")
    parser.add_argument("--max-gap", type=float, default=2.0, help="gaps narrower than this are thin (units)")
    parser.add_argument("--area-tol", type=float, default=1.0, help="ignore thin gaps smaller than this (units^2)")
    parser.add_argument("--write-gaps", type=Path, help="write the thin-gap triangles to this STL, for viewing")
    args = parser.parse_args(argv)

    geometry = GeometryConfig.from_toml(args.geometry)
    report, gap_faces = check_geometry(geometry, args.angles, args.flush_tol, args.max_gap, args.area_tol)
    print(format_report(report))
    if args.write_gaps and gap_faces:
        write_ascii_stl(args.write_gaps, gap_faces)
        print(f"wrote thin-gap triangles to {args.write_gaps}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
