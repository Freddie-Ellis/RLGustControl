"""Build an OpenFOAM case that meshes one geometry with snappyHexMesh.

`write_mesh_case` writes everything the HPC job (hpc/mesh_case.slurm) needs to mesh one set of deflections:

    constant/triSurface/aircraft.stl   the whole aircraft in metres, control surfaces deflected, with one
                                       named solid per body; each solid becomes a wall patch of that name
    constant/triSurface/aircraft_features.obj   its sharp edges (chosen in skin.py), read by snappyHexMesh
                                       for feature refinement and snapping
    system/blockMeshDict               background box of uniform hex cells around the aircraft
    system/snappyHexMeshDict           refinement, snapping and prism-layer settings
    system/meshQualityDict             quality limits snappyHexMesh must respect while moving points
    system/meshSlices                  cutting planes + wall surfaces written as small VTK files to download
    system/controlDict, fvSchemes, fvSolution, decomposeParDict   required by the OpenFOAM utilities
    case.toml                          copy of the case file the case was built from
    mesh_case.json                     every setting used, the job id and when it was written; read back by
                                       mesh_report.py

How snappyHexMesh builds the mesh, and how the settings map onto it:
    1. Castellation: start from the background cells (size `base_cell_size`) and split them in half
       repeatedly near the surfaces. A cell at refinement level n has size base_cell_size / 2**n.
       Cells inside the bodies are removed, keeping only those connected to `locationInMesh`.
    2. Snapping: move the jagged cell faces onto the STL surfaces and their feature edges.
    3. Layers: push the mesh away from the walls and insert thin prism cells to resolve the boundary layer.

Before writing the STL, the touching bodies are merged into the aircraft's outer skin (their boolean union,
see skin.py), so that faces hidden between bodies don't trigger refinement along every seam. The skin is checked
first (closed, no slits narrower than skin_min_gap) and the case is not written if it fails.

Run as a script (normally from the Slurm job), with a case file from configs/cases/:
    python -m src.cfd.mesh configs/cases/uav_wingtip_10.toml hpc/runs/uav_wingtip_10/mesh --n-procs 32
"""

import argparse
import json
import math
import os
import shutil
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

from src.cfd.case import CaseSpec
from src.cfd.deflect import deflected_bodies
from src.cfd.foam import foam_vector, write_foam_file
from src.cfd.geometry import GeometryConfig, StrictModel
from src.cfd.skin import check_skin, outer_skin
from src.cfd.stl import write_ascii_stl, write_obj_edges

SURFACE = "aircraft"  # name of the combined STL and of its snappyHexMesh geometry entry
FEATURES = f"{SURFACE}_features.obj"  # its sharp edges, chosen in skin.py, in constant/triSurface

type Vec3 = tuple[float, float, float]


class MeshAcceptance(StrictModel):
    """Limits a finished mesh must meet, checked by mesh_report.py after checkMesh."""
    max_non_orthogonality: float = 70.0      # degrees; above ~70 the solver needs extra correctors
    max_skewness: float = 4.0                # checkMesh's own limit for internal faces
    min_layer_thickness_percent: float = 80.0  # warn if a wall gets less than this share of its layer thickness


class MeshSettings(StrictModel):
    """Meshing settings, loaded from a TOML file (see configs/mesh/). Lengths are in metres."""
    # Domain extent, in multiples of the aircraft's length (its x extent).
    upstream_lengths: float = Field(default=20.0, gt=0)
    downstream_lengths: float = Field(default=40.0, gt=0)
    lateral_lengths: float = Field(default=20.0, gt=0)  # to each side, above and below

    # Background cell size; refinement level n gives cells of base_cell_size / 2**n.
    base_cell_size: float = Field(default=0.384, gt=0)

    # Surface refinement as (min, max) levels: min everywhere on the surface, max where it curves sharply
    # (by more than resolve_feature_angle between neighbouring cells, e.g. at leading edges).
    fixed_body_level: tuple[int, int] = (8, 9)
    control_surface_level: tuple[int, int] = (8, 9)
    feature_level: int = 9                 # level along extracted sharp edges
    feature_included_angle: float = 150.0  # edges where the surface turns by more than 180 - this are features
    resolve_feature_angle: float = 30.0
    n_cells_between_levels: int = 3        # buffer cells between refinement levels, for smooth size changes

    # Volume refinement boxes, levels as above. The near box surrounds the aircraft (padded by
    # near_padding_lengths aircraft lengths); the wake box runs wake_lengths aircraft lengths downstream from
    # the trailing edge.
    near_level: int = 5
    near_padding_lengths: float = 0.3
    wake_level: int = 5
    wake_lengths: float = 2.0

    # Range of flow angles (deg) the mesh must serve; one mesh is solved at every angle in it. The wake leaves
    # along the freestream, so the wake box is widened until the wake stays inside it at every angle: upward
    # for positive angle of attack, downward for negative, and to both sides for sideslip up to +-max_beta_deg.
    min_alpha_deg: float = Field(default=-10.0, ge=-60, le=0)
    max_alpha_deg: float = Field(default=30.0, ge=0, le=60)
    max_beta_deg: float = Field(default=10.0, ge=0, le=60)

    # Prism layers on every wall. Thicknesses are relative to the local surface cell size.
    n_layers: int = 5
    layer_expansion_ratio: float = 1.2
    final_layer_thickness: float = 0.5
    min_layer_thickness: float = 0.1
    n_grow: int = Field(default=0, ge=0)  # where layers collapse, also stop them over this many rings of faces
    # How snappyHexMesh moves the mesh aside for the layers: "medial_axis" (its default) or "displacement"
    # (displacementMotionSolver, a Laplacian solve: slower, often better coverage on curves and junctions).
    layer_shrinker: Literal["medial_axis", "displacement"] = "medial_axis"

    # Iteration counts. More iterations rarely improve the result once it settles, but cost run time.
    n_snap_solve_iter: int = 100     # snapping: mesh-displacement smoothing iterations (nSolveIter)
    n_layer_iter: int = 50           # layer addition: overall iterations (nLayerIter)
    n_relaxed_layer_iter: int = 20   # ...after which the relaxed quality limits apply (nRelaxedIter)

    # The skin must have no slits narrower than this (m): far too thin to mesh, they would only attract
    # refinement. Checked before writing the case (see skin.check_skin).
    skin_min_gap: float = Field(default=1e-4, gt=0)

    max_local_cells: int = 5_000_000     # per processor, during castellation
    max_global_cells: int = 100_000_000  # whole mesh; castellation stops refining beyond this

    acceptance: MeshAcceptance = MeshAcceptance()

    @classmethod
    def from_toml(cls, path: Path, overrides: dict[str, Any] | None = None) -> MeshSettings:
        """Load settings from a TOML file, with any `overrides` (e.g. a case's [mesh_overrides]) replacing
        the file's values. Overrides are checked like the file, so a misspelt name is an error."""
        with open(path, "rb") as f:
            return cls.model_validate({**tomllib.load(f), **(overrides or {})})

    def cell_size(self, level: int) -> float:
        return self.base_cell_size / 2**level


class Box(StrictModel):
    lo: Vec3
    hi: Vec3


def _domain(lo: torch.Tensor, hi: torch.Tensor, s: MeshSettings) -> tuple[Box, tuple[int, int, int]]:
    """Background box around the aircraft bounding box [lo, hi], sized to a whole number of base cells.

    Spanwise (y) the box is centred on y = 0 with an even number of cells, so a cell face lies on the
    centreline and the background mesh is its own mirror image. Mirror-image geometries (and the left and
    right halves of a symmetric one) are then refined identically, rather than differently depending on where
    the cells happen to fall relative to each side.
    """
    length = float(hi[0] - lo[0])
    pad_lo = torch.tensor([s.upstream_lengths, s.lateral_lengths, s.lateral_lengths], dtype=torch.float64)
    pad_hi = torch.tensor([s.downstream_lengths, s.lateral_lengths, s.lateral_lengths], dtype=torch.float64)
    d_lo = lo - pad_lo * length
    n_cells = torch.ceil((hi + pad_hi * length - d_lo) / s.base_cell_size).to(torch.int64)
    half_ny = math.ceil((max(-float(lo[1]), float(hi[1])) + s.lateral_lengths * length) / s.base_cell_size)
    d_lo[1] = -half_ny * s.base_cell_size
    n_cells[1] = 2 * half_ny
    d_hi = d_lo + n_cells * s.base_cell_size
    nx, ny, nz = (int(n) for n in n_cells)
    return Box(lo=tuple(d_lo.tolist()), hi=tuple(d_hi.tolist())), (nx, ny, nz)


def _location_in_mesh(lo: torch.Tensor, hi: torch.Tensor, domain: Box, s: MeshSettings) -> Vec3:
    """A point in the fluid, one aircraft length upstream. It is nudged off cell faces at every refinement
    level, since snappyHexMesh needs it strictly inside a cell."""
    target = torch.tensor([float(lo[0] - (hi[0] - lo[0])), float(lo[1] + hi[1]) / 2, float(lo[2] + hi[2]) / 2])
    origin = torch.tensor(domain.lo)
    cells = torch.floor((target - origin) / s.base_cell_size)
    point = origin + (cells + torch.tensor([0.4372, 0.4519, 0.4683])) * s.base_cell_size
    x, y, z = point.tolist()
    return (x, y, z)


def _slice_positions(geometry: GeometryConfig, bodies: dict[str, torch.Tensor]) -> list[float]:
    """Spanwise (y) positions for cutting planes: the root, plus the middle of each starboard control surface."""
    positions = [0.0]
    for surface in geometry.control_surfaces:
        y = bodies[surface.name][..., 1]
        middle = float(y.min() + y.max()) / 2
        if middle > 0 and all(abs(middle - p) > 0.01 for p in positions):
            positions.append(round(middle, 4))
    return sorted(positions)


def write_mesh_case(
    geometry: GeometryConfig,
    settings: MeshSettings,
    deflections_deg: dict[str, float],
    case_dir: Path,
    n_procs: int,
) -> dict[str, object]:
    """Write a complete, ready-to-mesh OpenFOAM case for one set of control surface deflections.

    Returns the case summary that is also saved as mesh_case.json.
    """
    s = settings
    bodies = deflected_bodies(geometry, deflections_deg)  # metres, each body closed
    skin = outer_skin(bodies)
    problems, _ = check_skin(skin.patches, s.skin_min_gap)
    if problems:
        raise ValueError("the aircraft's skin is not fit to mesh (fix the CAD model):\n" + "\n".join(problems))
    write_ascii_stl(case_dir / "constant" / "triSurface" / f"{SURFACE}.stl", skin.patches)
    # Sharp edges, for snappyHexMesh to refine along and snap to. They are chosen in skin.py rather than by
    # surfaceFeatureExtract, which would also pick up the tiny steps along every seam between bodies, and
    # snappyHexMesh reads the edge file directly.
    write_obj_edges(case_dir / "constant" / "triSurface" / FEATURES,
                    skin.feature_edges(180.0 - s.feature_included_angle))

    corners = torch.cat([t.reshape(-1, 3) for t in bodies.values()])
    lo, hi = corners.min(dim=0).values, corners.max(dim=0).values
    # Make the box symmetric about y = 0 (see _domain), so the refinement boxes are symmetric too.
    half_span = max(-float(lo[1]), float(hi[1]))
    lo[1], hi[1] = -half_span, half_span
    length = float(hi[0] - lo[0])
    domain, (nx, ny, nz) = _domain(lo, hi, s)
    pad = s.near_padding_lengths * length
    near = Box(lo=tuple((lo - pad).tolist()), hi=tuple((hi + pad).tolist()))
    # Over the wake box's length the wake drifts sideways by length * tan(angle): up at positive alpha
    # (the freestream then comes from below the nose), down at negative alpha, sideways with sideslip.
    reach = s.wake_lengths * length
    rise = reach * math.tan(math.radians(s.max_alpha_deg))
    fall = reach * math.tan(math.radians(-s.min_alpha_deg))
    drift = reach * math.tan(math.radians(s.max_beta_deg))
    wake = Box(
        lo=(float(hi[0]) - 0.1 * length, near.lo[1] - drift, near.lo[2] - fall),
        hi=(float(hi[0]) + reach, near.hi[1] + drift, near.hi[2] + rise),
    )
    location = _location_in_mesh(lo, hi, domain, s)
    fixed_names = [b.name for b in geometry.fixed_bodies]
    surface_names = [c.name for c in geometry.control_surfaces]
    wall_regex = "(" + "|".join(bodies) + ")"
    shrinker = "" if s.layer_shrinker == "medial_axis" else f"""
    meshShrinker        displacementMotionSolver;
    solver              displacementLaplacian;
    displacementLaplacianCoeffs {{ diffusivity quadratic inverseDistance ({" ".join(bodies)}); }}"""

    # Background mesh: one block of uniform hex cells. All six faces form one far-field patch, "freestream":
    # the solver's freestream conditions make each part of it an inlet or an outlet depending on whether the
    # flow enters or leaves there, so the same mesh serves every angle of attack and sideslip.
    (x0, y0, z0), (x1, y1, z1) = domain.lo, domain.hi
    write_foam_file(case_dir, "system/blockMeshDict", f"""
scale 1;

vertices
(
    ({x0:.6g} {y0:.6g} {z0:.6g}) ({x1:.6g} {y0:.6g} {z0:.6g}) ({x1:.6g} {y1:.6g} {z0:.6g}) ({x0:.6g} {y1:.6g} {z0:.6g})
    ({x0:.6g} {y0:.6g} {z1:.6g}) ({x1:.6g} {y0:.6g} {z1:.6g}) ({x1:.6g} {y1:.6g} {z1:.6g}) ({x0:.6g} {y1:.6g} {z1:.6g})
);

blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );

boundary
(
    freestream {{ type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
""")

    # Each named solid in aircraft.stl is a "region" of the surface. Naming each region after its body makes
    # the patch take the body's name (otherwise it would be aircraft_<body>).
    region_names = "\n".join(f"            {name} {{ name {name}; }}" for name in bodies)

    def region_entry(name: str, levels: tuple[int, int]) -> str:
        return f"            {name} {{ level ({levels[0]} {levels[1]}); patchInfo {{ type wall; }} }}"

    region_levels = "\n".join(
        [region_entry(n, s.fixed_body_level) for n in fixed_names]
        + [region_entry(n, s.control_surface_level) for n in surface_names]
    )

    write_foam_file(case_dir, "system/snappyHexMeshDict", f"""
// Cell sizes: level 0 = {s.cell_size(0):.4g} m, near box (level {s.near_level}) = {s.cell_size(s.near_level) * 1e3:.3g} mm,
// fixed bodies = {s.cell_size(s.fixed_body_level[0]) * 1e3:.3g}-{s.cell_size(s.fixed_body_level[1]) * 1e3:.3g} mm,
// control surfaces = {s.cell_size(s.control_surface_level[0]) * 1e3:.3g}-{s.cell_size(s.control_surface_level[1]) * 1e3:.3g} mm,
// feature edges = {s.cell_size(s.feature_level) * 1e3:.3g} mm.

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    {SURFACE}
    {{
        type triSurfaceMesh;
        file "{SURFACE}.stl";
        regions
        {{
{region_names}
        }}
    }}
    near_box {{ type searchableBox; min {foam_vector(near.lo)}; max {foam_vector(near.hi)}; }}
    wake_box {{ type searchableBox; min {foam_vector(wake.lo)}; max {foam_vector(wake.hi)}; }}
}}

castellatedMeshControls
{{
    maxLocalCells       {s.max_local_cells};
    maxGlobalCells      {s.max_global_cells};
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels {s.n_cells_between_levels};

    features
    (
        {{ file "{FEATURES}"; level {s.feature_level}; }}
    );

    // Every body (region) becomes its own wall patch, named after the body, with its own refinement level.
    refinementSurfaces
    {{
        {SURFACE}
        {{
            level ({s.fixed_body_level[0]} {s.fixed_body_level[1]});
            patchInfo {{ type wall; }}
            regions
            {{
{region_levels}
            }}
        }}
    }}
    resolveFeatureAngle {s.resolve_feature_angle:g};

    refinementRegions
    {{
        near_box {{ mode inside; levels ((1e15 {s.near_level})); }}
        wake_box {{ mode inside; levels ((1e15 {s.wake_level})); }}
    }}

    // A point in the fluid: the mesh is kept on this side of the surfaces.
    locationInMesh {foam_vector(location)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      {s.n_snap_solve_iter};
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap    false;
    explicitFeatureSnap    true;   // snap to the edges in the features file
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;   // thicknesses below are fractions of the local surface cell size
    layers
    {{
        "{wall_regex}" {{ nSurfaceLayers {s.n_layers}; }}
    }}
    expansionRatio      {s.layer_expansion_ratio:g};
    finalLayerThickness {s.final_layer_thickness:g};
    minThickness        {s.min_layer_thickness:g};
    nGrow               {s.n_grow};
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          {s.n_layer_iter};
    nRelaxedIter        {s.n_relaxed_layer_iter};{shrinker}
}}

meshQualityControls
{{
    #include "meshQualityDict"
    // Looser limit allowed after nRelaxedIter layer iterations; kept at the acceptance limit so snappyHexMesh
    // never produces a mesh that mesh_report.py would then fail.
    relaxed {{ maxNonOrtho {s.acceptance.max_non_orthogonality:g}; }}
}}

writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
debug 0;
""")

    # Quality limits snappyHexMesh enforces while snapping and adding layers (the OpenFOAM defaults).
    write_foam_file(case_dir, "system/meshQualityDict", """
maxNonOrtho          65;
maxBoundarySkewness  20;
maxInternalSkewness  4;
maxConcave           80;
minVol               1e-13;
minTetQuality        1e-15;
minArea              -1;
minTwist             0.02;
minDeterminant       0.001;
minFaceWeight        0.05;
minVolRatio          0.01;
minTriangleTwist     -1;
nSmoothScale         4;
errorReduction       0.75;
""")

    # Small extracts to download and view in ParaView, since the full mesh is too large for a laptop:
    # cutting planes through the root and each starboard control surface, and the wall surfaces.
    # They carry the quality fields written by `checkMesh -writeAllFields` (cell edges are kept visible).
    planes = "\n".join(f"""
        y_{y * 1e3:.0f}mm
        {{
            type cuttingPlane;
            planeType pointAndNormal;
            pointAndNormalDict {{ point (0 {y:.6g} 0); normal (0 1 0); }}
            interpolate false;
        }}""" for y in _slice_positions(geometry, bodies))
    write_foam_file(case_dir, "system/meshSlices", f"""
functions
{{
    meshSlices
    {{
        type            surfaces;
        libs            (sampling);
        surfaceFormat   vtk;
        interpolationScheme cell;
        // nSurfaceLayers and thicknessFraction (achieved / requested layer thickness) are snappyHexMesh's
        // layer fields, only non-zero on the walls; fields that are missing are skipped.
        fields          (cellVolume cellAspectRatio nonOrthoAngle skewness cellShapes nSurfaceLayers thicknessFraction);
        surfaces
        {{
{planes}
            walls {{ type patch; patches ("{wall_regex}"); interpolate false; }}
        }}
    }}
}}
""")

    write_foam_file(case_dir, "system/controlDict", """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
runTimeModifiable false;
""")
    # snappyHexMesh needs these two files to exist; the solver stage will replace them.
    write_foam_file(case_dir, "system/fvSchemes", """
ddtSchemes           { default steadyState; }
gradSchemes          { default Gauss linear; }
divSchemes           { default none; }
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
wallDist             { method meshWave; }
""")
    # The displacement layer shrinker solves for the mesh motion, so it needs a linear solver for it.
    displacement_solver = ("cellDisplacement { solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.01; }"
                           if s.layer_shrinker == "displacement" else "")
    write_foam_file(case_dir, "system/fvSolution", f"solvers {{ {displacement_solver} }}")
    write_foam_file(case_dir, "system/decomposeParDict", f"numberOfSubdomains {n_procs};\nmethod scotch;")

    info: dict[str, object] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "geometry": geometry.name,
        "deflections_deg": deflections_deg,
        "n_procs": n_procs,
        "aircraft_bbox_m": {"lo": lo.tolist(), "hi": hi.tolist()},
        "domain_m": domain.model_dump(),
        "background_cells": [nx, ny, nz],
        "location_in_mesh": location,
        "settings": settings.model_dump(),
    }
    (case_dir / "mesh_case.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write an OpenFOAM case that meshes one geometry.")
    parser.add_argument("case_file", type=Path, help="case TOML (see configs/cases/)")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--n-procs", type=int, default=1, help="number of MPI ranks the mesh will be built on")
    args = parser.parse_args(argv)

    case = CaseSpec.from_toml(args.case_file)
    settings = MeshSettings.from_toml(case.mesh, case.mesh_overrides)
    write_mesh_case(case.load_geometry(), settings, case.deflections_deg, args.case_dir, args.n_procs)
    shutil.copyfile(args.case_file, args.case_dir / "case.toml")
    print(f"wrote mesh case to {args.case_dir}")
    print(f"  {case.description or args.case_file.stem}; deflections: {case.deflections_deg or 'all zero'}")
    print(f"  smallest surface cells: {settings.cell_size(settings.fixed_body_level[1]) * 1e3:.3g} mm on fixed "
          f"bodies, {settings.cell_size(settings.control_surface_level[1]) * 1e3:.3g} mm on control surfaces")


if __name__ == "__main__":
    main()
