"""Merge the touching bodies of the aircraft into its outer (wetted) skin before meshing, and check it.

The aircraft is modelled as separate closed bodies that touch or overlap: a flap's nose is buried in the main
body, and its end faces sit flush against it. snappyHexMesh would see those hidden faces too. Two faces lying
on top of each other with opposite normals look like extremely sharp curvature, so it refines around them to
the maximum level, and the refinement spills out onto the outer skin as dense bands along every seam.

`outer_skin` replaces the bodies with their boolean union, computed with the manifold3d library. The union
cuts triangles along the curves where bodies meet and keeps only the parts on the outside, so the result is a
single closed surface with no hidden faces. Each triangle of the union remembers which body it came from, so
each body still becomes its own patch (for per-surface forces and hinge moments).

`Skin.feature_edges` lists the sharp edges for snappyHexMesh to refine along and snap to. They are not
found from the merged surface alone: where two bodies' skins meet at a seam, their separate triangulations
differ by up to the STL export deviation, so the union leaves a tiny step (a sliver of the otherwise hidden
flap end face or nose) all along the seam. Each step looks like a sharp edge, and snapping to it dents the
surface. So an edge only counts if both of its faces come from source triangles that are mostly on the
outside, which every step sliver fails.

For this to work, bodies must overlap or share flat faces. Two separately triangulated copies of the same
curved surface (a flap nose in a zero-clearance cove) weave through each other by up to the STL export
deviation, and the union then keeps slivers of both. `check_skin` finds such problems, and any other narrow
slit, so they can be fixed in the CAD model rather than patched here.

Run as a script to write the skin and check it, e.g.:
    uv run python -m src.cfd.skin cad/uav/geometry.toml out/skin_zero_mm.stl --cad-units
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import manifold3d
import numpy as np
import torch

from src.cfd.deflect import deflected_bodies, parse_deflections
from src.cfd.geometry import GeometryConfig
from src.cfd.stl import Triangles, facet_normals, write_ascii_stl, write_obj_edges
from src.cfd.surface import directed_edges, open_edge_count, weld


def _to_manifold(triangles: Triangles, name: str) -> manifold3d.Manifold:
    """Convert a closed surface to a manifold3d solid, tagged with its own original id."""
    vertices, faces = weld(triangles)
    mesh = manifold3d.Mesh64(vert_properties=vertices.numpy(), tri_verts=faces.numpy().astype(np.uint32))
    solid = manifold3d.Manifold(mesh)
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f"{name} is not a closed, consistently oriented surface: {solid.status()}")
    return solid.as_original()


@dataclass(frozen=True)
class Skin:
    """The aircraft's merged outer skin."""
    patches: dict[str, Triangles]  # the skin's triangles, by the body they came from (its patch)
    exposed: torch.Tensor          # per triangle, in patch order: its source triangle is mostly on the outside

    def triangles(self) -> Triangles:
        return torch.cat(list(self.patches.values()))

    def feature_edges(self, min_angle_deg: float) -> torch.Tensor:
        """Sharp edges of the skin, shape (k, 2, 3): edges where the surface turns by more than
        `min_angle_deg`, between two faces whose source triangles are mostly on the outside."""
        triangles = self.triangles()
        vertices, faces = weld(triangles)
        edges = directed_edges(faces).sort(dim=1).values    # (3n, 2): face i's edges are rows i, n + i, 2n + i
        face_of_edge = torch.arange(len(faces)).repeat(3)
        unique_edges, edge_number, uses = torch.unique(edges, dim=0, return_inverse=True, return_counts=True)

        # The two faces on each edge used exactly twice (edges used four times, where two parts of the skin
        # touch along a line, are left out).
        order = edge_number.argsort(stable=True)
        first = torch.cumsum(uses, dim=0) - uses
        shared = (uses == 2).nonzero().flatten()
        face_a, face_b = face_of_edge[order[first[shared]]], face_of_edge[order[first[shared] + 1]]

        normals = facet_normals(triangles)
        sharp = (normals[face_a] * normals[face_b]).sum(dim=1) < math.cos(math.radians(min_angle_deg))
        keep = sharp & self.exposed[face_a] & self.exposed[face_b]
        # Take the coordinates from the triangles' own corners rather than the welded (rounded) vertices, so
        # the edge ends are exactly the skin's corner points.
        corners = torch.zeros_like(vertices)
        corners[faces.flatten()] = triangles.reshape(-1, 3)
        return corners[unique_edges[shared[keep]]]


def outer_skin(bodies: dict[str, Triangles]) -> Skin:
    """The outer skin of the union of `bodies` (each closed), split back into the faces of each body.

    Every body keeps its key, with the triangles of the union that lie on its surface (none if it is hidden
    entirely). Where two bodies' faces coincide exactly, the union keeps one of them.
    """
    solids: list[manifold3d.Manifold] = []
    name_of_id: dict[int, str] = {}
    for name, triangles in bodies.items():
        solid = _to_manifold(triangles, name)
        name_of_id[solid.original_id()] = name
        solids.append(solid)
    union = manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add)
    if union.status() != manifold3d.Error.NoError:
        raise ValueError(f"union of the bodies failed: {union.status()}")

    mesh = union.to_mesh64()
    vertices = torch.from_numpy(np.asarray(mesh.vert_properties)[:, :3].copy())
    faces = torch.from_numpy(np.asarray(mesh.tri_verts).astype(np.int64))

    # Triangles come grouped in runs, one per source body: run k is triangles run_index[k] / 3 up to
    # run_index[k + 1] / 3, from the body whose original id is run_original_id[k].
    # face_id[i] is the number of triangle i's source triangle within its body.
    run_start = torch.from_numpy(np.asarray(mesh.run_index).astype(np.int64)) // 3
    run_id = np.asarray(mesh.run_original_id)
    body_of_face = torch.repeat_interleave(torch.arange(len(run_id)), run_start[1:] - run_start[:-1])
    source = torch.from_numpy(np.asarray(mesh.face_id).astype(np.int64))

    # Share of each source triangle that ends up on the skin: the area of the union triangles cut from it,
    # over its own area. Source triangles are numbered across all bodies in turn, from first_source.
    names = [name_of_id[int(original_id)] for original_id in run_id]
    first_source = dict(zip(bodies, np.cumsum([0] + [len(t) for t in bodies.values()]).tolist()))
    global_source = source + torch.tensor([first_source[name] for name in names])[body_of_face]
    source_area = torch.cat([_areas(t) for t in bodies.values()])
    kept_area = torch.zeros_like(source_area).index_add_(0, global_source, _areas(vertices[faces]))
    exposed = (kept_area / source_area.clamp_min(1e-300))[global_source] > 0.5

    # Where faces of two bodies coincide (flush flat faces), the union also produces tiny pieces enclosing
    # no volume, and closed pockets between bodies become separate pieces too. Neither is part of the
    # aircraft's outer surface, so keep only the largest piece.
    keep = _largest_piece(vertices, faces)
    faces, body_of_face, exposed = faces[keep], body_of_face[keep], exposed[keep]

    patches = {name: torch.zeros((0, 3, 3), dtype=torch.float64) for name in bodies}
    patch_exposed = {name: torch.zeros(0, dtype=torch.bool) for name in bodies}
    for k, name in enumerate(names):
        in_run = body_of_face == k
        patches[name] = torch.cat([patches[name], vertices[faces[in_run]]])
        patch_exposed[name] = torch.cat([patch_exposed[name], exposed[in_run]])
    return Skin(patches=patches, exposed=torch.cat(list(patch_exposed.values())))


def _areas(triangles: Triangles) -> torch.Tensor:
    """Area of each triangle."""
    return torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]).norm(dim=1) / 2


def _largest_piece(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Mask of the faces in the connected piece of the surface that encloses the largest volume.

    Pieces are found by label propagation: every vertex starts with its own number, then repeatedly takes
    the smallest number among the triangles it belongs to, until nothing changes. Each piece ends up
    labelled with its smallest vertex number. (manifold3d's decompose() does the same but takes minutes
    when there are thousands of pieces.)
    """
    labels = torch.arange(len(vertices))
    while True:
        face_min = labels[faces].min(dim=1).values
        new = labels.scatter_reduce(0, faces.flatten(), face_min.repeat_interleave(3), reduce="amin")
        new = new[new]  # pointer jumping: take the label of your label, which speeds up convergence
        if torch.equal(new, labels):
            break
        labels = new
    face_label = labels[faces[:, 0]]
    # Signed volume of each piece, from the divergence theorem: sum over its triangles of a . (b x c) / 6.
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face_volume = (a * torch.linalg.cross(b, c)).sum(dim=1) / 6
    piece_volume = torch.zeros(len(vertices), dtype=torch.float64).index_add_(0, face_label, face_volume)
    return face_label == piece_volume.argmax()


def narrow_gap_faces(skin: dict[str, Triangles], min_gap: float) -> dict[str, Triangles]:
    """Faces of the skin that look out across a gap narrower than `min_gap` onto another part of the skin.

    These are the walls of slits and slivers too thin to mesh, which snappyHexMesh would refine around. A ray
    is cast from the centre of each face along its outward normal; the face is reported if it hits the skin
    again within `min_gap`.
    """
    # Work in units of min_gap: manifold3d's ray cast has fixed tolerances, so the same check in metres and
    # in millimetres would otherwise give different answers.
    triangles = torch.cat(list(skin.values())) / min_gap
    vertices, faces = weld(triangles)
    solid = manifold3d.Manifold(
        manifold3d.Mesh64(vert_properties=vertices.numpy(), tri_verts=faces.numpy().astype(np.uint32))
    )
    centres = triangles.mean(dim=1)
    normals = facet_normals(triangles)
    start = (centres + 1e-6 * normals).tolist()  # just off the face, so it doesn't hit itself
    end = (centres + normals).tolist()
    narrow = torch.tensor([bool(solid.ray_cast(s, e)) for s, e in zip(start, end)], dtype=torch.bool)

    found: dict[str, Triangles] = {}
    first = 0
    for name, body in skin.items():
        hits = body[narrow[first:first + len(body)]]
        if len(hits):
            found[name] = hits
        first += len(body)
    return found


def check_skin(skin: dict[str, Triangles], min_gap: float) -> tuple[list[str], dict[str, Triangles]]:
    """Problems with the skin that would spoil the mesh, as messages (empty if none), and the narrow-gap faces.

    Checks that the skin is closed (snappyHexMesh would mesh the inside of the aircraft through any hole) and
    that no part of it faces another across a gap narrower than `min_gap` (see narrow_gap_faces). A body's
    narrow-gap faces are only a problem if they add up to more than (10 min_gap)^2 (1 mm^2 for a 0.1 mm gap):
    the union leaves a few tiny slivers along sharp creases and seams, far too small to affect the mesh.
    """
    problems: list[str] = []
    open_edges = open_edge_count(torch.cat(list(skin.values())))
    if open_edges:
        problems.append(f"skin is not closed: {open_edges} open edges")

    narrow = narrow_gap_faces(skin, min_gap)
    for name, faces_found in narrow.items():
        corners = faces_found.reshape(-1, 3)
        area = float(_areas(faces_found).sum())
        if area <= (10 * min_gap) ** 2:
            continue
        lo = [round(x, 4) for x in corners.min(dim=0).values.tolist()]
        hi = [round(x, 4) for x in corners.max(dim=0).values.tolist()]
        problems.append(f"{name}: {len(faces_found)} faces ({area:.3g} area) face the skin across a gap "
                        f"narrower than {min_gap:g}, within box {lo} to {hi}")
    return problems, narrow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the aircraft's outer skin as one STL and check it.")
    parser.add_argument("geometry", type=Path, help="path to geometry.toml")
    parser.add_argument("out_file", type=Path, help="STL file to write (a named solid per body)")
    parser.add_argument("-d", "--deflect", nargs="*", default=[], metavar="NAME=DEG",
                        help="control surface deflections in degrees; unlisted surfaces stay at 0")
    parser.add_argument("--cad-units", action="store_true",
                        help="write in the model's units (e.g. mm) instead of metres, to overlay on the CAD model")
    parser.add_argument("--min-gap", type=float, default=0.1,
                        help="report gaps narrower than this, in mm (default 0.1)")
    parser.add_argument("--feature-angle", type=float, default=30.0,
                        help="sharp edges turn by more than this, in degrees (default 30)")
    args = parser.parse_args(argv)

    geometry = GeometryConfig.from_toml(args.geometry)
    deflections = parse_deflections(args.deflect)
    bodies = deflected_bodies(geometry, deflections, in_metres=not args.cad_units)
    skin = outer_skin(bodies)
    write_ascii_stl(args.out_file, skin.patches)
    features_file = args.out_file.with_name(args.out_file.stem + "_features.obj")
    write_obj_edges(features_file, skin.feature_edges(args.feature_angle))
    print(f"wrote {args.out_file} with deflections {deflections or 'all zero'}, and its sharp edges to {features_file}")

    min_gap = args.min_gap * 1e-3 / (geometry.scale_to_m if args.cad_units else 1.0)  # mm -> output units
    problems, narrow = check_skin(skin.patches, min_gap)
    if problems and narrow:
        narrow_file = args.out_file.with_name(args.out_file.stem + "_narrow_gaps.stl")
        write_ascii_stl(narrow_file, narrow)
        print(f"wrote the narrow-gap faces to {narrow_file}")
    print("\n".join(["FAILED:"] + problems) if problems else "OK: skin is closed and has no narrow gaps")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
