# CFD pipeline

The control model is trained on a surrogate of the delta wing UAV's aerodynamics, built from a dataset of RANS CFD
solutions over a range of flow angles and control surface deflections. The pipeline is scripted end to end and runs on
the Iridis 6 HPC: case files go in, and coefficients, hinge moments and surface pressures come out.

```
SolidWorks ─► one STL per body ─► deflect control surfaces ─► merge into one outer skin, check it
    ─► snappyHexMesh (once per set of deflections) ─► simpleFoam, k-ω SST (once per α, β) ─► results.json
    ─► postpro/ tables and plots ─► (planned) Parquet dataset ─► surrogate
```

**Status:** geometry, meshing, solving and post-processing work on Iridis. The mesh-independence study has been run on
the pitched background mesh; α = 20° showed a jump in C_L between the medium and fine meshes, which the vortex study
(`configs/cases/vortex_study/`) is investigating. The actuator authority batch (`configs/cases/batch2`) is set up and
waits on the mesh choice. Collecting results into a dataset is planned.

## Code

| Module | Purpose |
|---|---|
| `src/cfd/geometry.py` | Loads and validates `cad/<model>/geometry.toml`: bodies, hinges, deflection limits, reference values |
| `src/cfd/stl.py` | Reads ASCII and binary STL; writes multi-solid ASCII STL and OBJ edge files |
| `src/cfd/surface.py` | Welds triangles into an indexed mesh and counts open edges |
| `src/cfd/deflect.py` | Loads the bodies and rotates control surfaces about their hinges |
| `src/cfd/skin.py` | Merges the bodies into one closed outer skin (boolean union), picks its feature edges, checks it |
| `src/cfd/case.py` | Loads a case file: geometry, mesh and solver settings, deflections, flow conditions |
| `src/cfd/foam.py` | Helpers for writing OpenFOAM input files |
| `src/cfd/mesh.py` | Writes a ready-to-mesh OpenFOAM case (skin STL, feature edges, every dictionary) |
| `src/cfd/mesh_report.py` | Reads the checkMesh and snappyHexMesh logs, applies the acceptance limits, writes `mesh_report.json` |
| `src/cfd/solve.py` | Writes a ready-to-run simpleFoam case for one flow condition on a finished mesh |
| `src/cfd/solve_report.py` | Turns the solver output into coefficients, hinge moments and a convergence verdict (`results.json`) |
| `src/cfd/submit.py` | Submits mesh and solve jobs to Slurm, skipping finished work (run through `hpc/submit_cases.sh`) |
| `src/cfd/runs.py` | Lists every run and its state, read from the run folders |
| `src/cfd/batch_cases.py` | Turns a table of cases (deflections, α, β) into case files, one per set of deflections |

| Post-processing (`postpro/`, run locally on downloaded run folders) | Purpose |
|---|---|
| `postpro/summary.py` | Tables of every set point's coefficients, hinge moments, α and β derivatives, slopes and control effects (change from the clean case); coefficient and hinge-moment plots against α |
| `postpro/convergence.py` | Convergence table (iterations, gap from the running mean, final ranges and residuals); coefficient and residual histories |
| `postpro/pressure.py` | Cp along chordwise cuts at spanwise stations, overlaid across runs, with the RMS difference from the last run |
| `postpro/boundary_layer.py` | y⁺ per patch and by area, prism-layer coverage, and skin friction along cuts against flat-plate correlations |
| `postpro/common.py` | Shared loading: run folders, `results.json`, the wall-surface VTK files, surface cuts |

| HPC file | Purpose |
|---|---|
| `hpc/setup_python.sh` | One-off: builds `.venv-hpc` (CPU PyTorch, Pydantic, manifold3d) on the login node |
| `hpc/submit_cases.sh` | Submits case files (wrapper around `src/cfd/submit.py`) |
| `hpc/mesh_case.slurm` | Mesh job: writes the case, blockMesh, snappyHexMesh, checkMesh, report |
| `hpc/solve_case.slurm` | Solve job: writes the case, potentialFoam, simpleFoam, report |
| `hpc/env.sh` | Shared setup for the above: loads OpenFOAM v2506 and the Python environment |

## Quick start
1. **Geometry** (only when the CAD changes): export each body as an STL into `cad/uav/` and check the file names in
   `cad/uav/geometry.toml` (see [Geometry](#1-geometry)).
2. **Case files**: write `configs/cases/<name>.toml` (see [Case files](#5-case-files-run-folders-and-submission)), or
   generate a folder of them from a table of cases:
   `.venv/Scripts/python.exe -m src.cfd.batch_cases <table.csv> configs/cases/<batch>`.
3. **On Iridis**: `git pull`, and after a dependency change `bash hpc/setup_python.sh`.
4. **Submit**: `bash hpc/submit_cases.sh configs/cases/<name>.toml`, or a folder of case files (options in
   [Submission](#submission)).
5. **Monitor**: `squeue -u $USER` lists jobs as `mesh_<case>` and `solve_<case>_<condition>`.
   `tail -f hpc/runs/<case>/mesh/slurm_<jobid>.out` follows a job.
6. **Check**: after `source hpc/env.sh && load_python`, run `python -m src.cfd.runs` for one line per case (`--failed`
   for only the failed meshes).
7. **Post-process** (locally): download the run folders and run the `postpro/` scripts (see
   [Post-processing](#6-post-processing)).

## 1. Geometry
- The aircraft is a SolidWorks multibody part. The CFD uses only the outer (wetted) surface. The main body and each
  control surface are separate bodies: the current model (`cad/uav/`) has a main body and six control surfaces per side
  (wingtip, leading-edge flap, outboard and inboard trailing-edge flaps, upper and lower splitter flaps).
- Each body is exported as its own STL, at zero deflection, in the SolidWorks part frame (x aft, y starboard, z up) in
  mm, and becomes its own OpenFOAM patch, so forces and hinge moments can be read per surface.
- `cad/<model>/geometry.toml` lists every body with its STL file, each control surface's hinge axis and deflection
  limits, and the reference point, chord, span and area used for the coefficients. Unknown keys, bad names and
  inconsistent limits are errors on load.
- The moment reference point is the estimated aerodynamic centre, (235.32, 0, −12.33) mm, on the wing's mid-plane. The
  wing is a flat plate at 3° incidence (trailing edge down), so α = −3° is flow-aligned. There is no CG yet; moments
  about any other point follow from M' = M + (r_ref − r') × F.

**STL export settings** (Save As → STL → Options): binary, mm, **Custom** resolution with deviation 0.04 mm (the
smallest SolidWorks allows) and angle 1°; tick **"Do not translate STL output data to positive space"** (otherwise the
hinge points no longer line up; `stl_offset` in `geometry.toml` can correct a shifted export); untick **"Save all
components of an assembly in a single file"**. Check the file names afterwards: SolidWorks sometimes changes the
`-3` / `-4` suffix.

**How bodies may meet.** The bodies are merged into one outer skin before meshing, which only gives a clean skin if they
**overlap, or meet on flat faces in a plane of constant x, y or z**. Avoid:
- a curved surface shared by two bodies with zero clearance, such as a flap nose in a cove of the same radius. The
  exports are triangulated separately and weave through each other, leaving slivers. Fill the cove so the nose is
  buried in the main body (done for every flap).
- flush faces at an angle to the axes, such as the swept leading-edge flap ends. STL stores single-precision
  coordinates, so the two copies end up a hair apart. Make the bodies overlap (the leading-edge flaps are 0.3 mm
  longer at each end).
- slits and near-misses, such as a loft ending just short of the wing. Extend one body into the other.

Every body must be closed (watertight); the current exports are. A body with a hole stops the mesh job at the skin step.

The left and right exports are triangulated slightly differently by SolidWorks, so the two sides of a symmetric mesh
can differ in small details (for example around the wingtip housings).

## 2. Control surfaces and the outer skin
Each hinge axis is given by two points, `p1` and `p2`. Positive deflection is a right-hand rotation about the axis from
`p1` to `p2` (Rodrigues' formula: `p' = a + (p−a)cosδ + (k × (p−a))sinδ + k(k·(p−a))(1−cosδ)`). Port surfaces are
mirror images of the starboard ones, and a mirror reverses the rotation sense, so each port hinge uses the mirrored
points in swapped order; the same positive angle then gives the same physical motion on both sides.

| Surface | Positive deflection |
|---|---|
| Wingtip, inboard and outboard trailing-edge flaps | trailing edge up |
| Leading-edge flap | leading edge down |
| Upper and lower splitter flaps (0 to 90°) | open (upper swings up, lower swings down) |

For each case the deflected bodies are merged into one closed surface, their boolean union (the `manifold3d` library),
which cuts the triangles where bodies meet and keeps only the outside, so no hidden faces are left for snappyHexMesh to
refine around. Each triangle remembers which body it came from, and the skin is written as `aircraft.stl` with one
named solid (patch) per body. The skin is then checked: it must be closed, and no part of it may face another across a
gap narrower than 0.1 mm. A failure names the bodies and the location. The skin and its feature edges are in each mesh
folder (`constant/triSurface/aircraft.stl`, `aircraft_features.obj`) for viewing in ParaView.

**Feature edges** (where snappyHexMesh refines and snaps to sharp edges) are chosen in `skin.py`, not by
`surfaceFeatureExtract`. Where two bodies meet, their separate triangulations leave a step of a few hundredths of a
millimetre along the seam, which a plain angle test sees as a sharp edge, refining every seam and denting the surface.
An edge counts only if it turns by more than 30° and both of its faces come from source triangles that are mostly on
the outside, which every seam step fails.

## 3. Meshing (snappyHexMesh)
- One mesh per set of deflections, reused for every flow condition. Settings are in `configs/mesh/uav.toml` (its
  comments give the cell size of each level); a case can change individual values with `[mesh_overrides]`.
- Refinement level n gives cells of `base_cell_size / 2^n`. With the medium `base_cell_size` of 0.384 m: 1.5 mm surface
  cells (level 8), refined to 0.75 mm at sharp curvature (leading edges) and along feature edges (level 9); 12 mm cells
  in a box around the aircraft and in the wake (level 5); 5 prism layers on every wall. Each cell takes the highest
  level any rule asks for.
- Around the skin, `surface_distance_levels = [[0.06, 7], [0.10, 6]]` adds 3 mm cells within 60 mm and 6 mm cells
  within 100 mm, to resolve the leading-edge shear layer and vortex (about 4.6M cells in all). It was chosen by the
  vortex study: without it, C_L at α = 20° was 0.820 against 0.90–0.93 with it, and the inboard upper-surface
  pressure, wanted for tap placement, kept changing up to this level. Distance refinement makes meshing slower, so
  allow `--mesh-time 04:00:00`.
- Background box about 20 aircraft lengths upstream, above, below and to each side, and 40 downstream, centred on the
  centreline with a cell face on y = 0, so a symmetric aircraft is meshed the same on both sides.
- The background box is pitched nose-up by `background_pitch_deg` (3°, the wing's incidence), so the flat wing
  surfaces lie parallel to the cells. Unpitched, a surface at 3° crosses a layer of cells every cell size / tan 3°
  (28.6 mm for 1.5 mm cells), and snapping leaves a dimple about 0.15 mm deep at each crossing, at every mesh size.
  Only the background cells are pitched: the geometry, flow angles and results stay in the model frame.
- Small dents (up to about 0.2–0.3 mm on the coarse mesh) remain on the tapered wingtips and along the fuselage–wing
  crease, where the surface still crosses the cells. More snapping relaxation (`n_snap_*`) made no difference and finer
  wingtip cells (`surface_levels`) only traded fewer, larger dents for more, smaller ones at twice the meshing time, so
  they are accepted.
- Optional settings: `surface_levels` (refinement levels per body), `surface_distance_levels` (refinement by distance
  from the skin, for the leading-edge vortex), and the snapping controls `n_snap_smooth_patch`, `n_snap_relax_iter`,
  `n_feature_snap_iter`, `snap_tolerance`.
- The six outer faces form one `freestream` patch, and the wake box is widened to contain the wake over the whole
  angle range in the mesh settings (`min_alpha_deg`, `max_alpha_deg`, `max_beta_deg`: currently α from −10° to 30°, β
  up to ±10°). So one mesh serves every flow direction in that range, and flow conditions outside it are rejected.
- The job finishes with `checkMesh` and `mesh_report.json`: cell counts, maximum non-orthogonality (limit 70°) and
  skewness (limit 4), and prism-layer coverage per wall (warning below 80%). The job fails if the mesh fails.
  `postProcessing/meshSlices/` holds cutting planes and the wall surfaces with quality fields, small enough to download
  and open in ParaView. Faces that checkMesh flags as bad are written as VTK under `postProcessing/constant/`.
- On a quiet node and 32 cores, the coarse (1.8M cells), medium (3.5M) and fine (6.5M) meshes take about 3, 6 and 14
  minutes. On a node shared with other users' jobs the same mesh can take 5–20 times longer (see
  [Submission](#submission)). snappyHexMesh does not speed up beyond about 32 cores at these sizes.

## 4. Solving (simpleFoam)
- Steady, incompressible RANS with the k-ω SST turbulence model; settings in `configs/solver/uav_rans.toml` (airspeed,
  air properties, freestream turbulence, iterations, convergence). **The airspeed is a placeholder (20 m/s) until the
  design cruise speed is set.**
- Walls use Spalding's wall function (`nutUSpaldingWallFunction`), valid across the y⁺ ≈ 10–30 that the prism layers
  give. The `freestream` patch uses `freestreamVelocity` / `freestreamPressure`, which act as inflow or outflow
  wherever the flow enters or leaves.
- Flow direction in the model frame (x aft, y starboard, z up): **U = V (cos α cos β, −sin β, sin α cos β)**. Positive
  α means the air comes from below the nose, positive β from the starboard side. Lift is measured perpendicular to U in
  the x-z plane, drag along U.
- Each condition starts from a potential-flow solution (`potentialFoam`) and runs simpleFoam until it converges (C_L,
  C_D and C_m each within 1e-4 of their 200-iteration running mean, checked by OpenFOAM's `runTimeControl`) or reaches
  `max_iterations` (3000). Conditions are independent, so they run as separate jobs in parallel. A solve takes about
  15, 30 and 60 minutes on the coarse, medium and fine meshes.
- **Results** (`results.json`, means over the last 200 iterations with their ranges): C_L, C_D, C_Y; C_l, C_m, C_n
  about the reference point in the usual flight-mechanics sense (right wing down, nose up, nose right positive); the
  hinge moment of each control surface about its own hinge axis (positive in the direction of positive deflection), in
  N·m and as C_h = M_h / (q S c); forces and moments in newtons; final residuals. A condition is converged if it met
  the stopping test above at its last iteration (the report recomputes it from the force history). An unconverged
  condition (for example unsteady separated flow at high α) is still reported, with its ranges showing how much it
  oscillated.
- Also written: the force, moment and residual histories in `postProcessing/` (including the force and moment on each
  control surface, `forces_<surface>/`), wall pressure, wall shear stress and y⁺ as VTK
  (`postProcessing/wallSurfaces/`), and slices of p, U, k and ν_t on 10 planes across x and 11 across y (one on the
  centreline) around the aircraft (`postProcessing/flowSlices/`, set by `n_slices_x` / `n_slices_y`). The final 3D
  fields (about 200 MB per condition) are kept unless `keep_fields = false`.

## 5. Case files, run folders and submission
A case file in `configs/cases/` (subfolders allowed) describes one set of deflections and the flow conditions to solve
it at. Its file name is the case name, which must be unique:

```toml
description = "Both wingtips at +10 deg"
geometry = "cad/uav/geometry.toml"
mesh = "configs/mesh/uav.toml"
solver = "configs/solver/uav_rans.toml"   # leave out solver and [flow] to mesh only
[deflections_deg]                         # surfaces not listed stay at 0
wingtip_r = 10.0
wingtip_l = 10.0
[mesh_overrides]                          # optional
base_cell_size = 0.288
[solver_overrides]                        # optional, e.g. drop the full 3D fields
keep_fields = false
[flow]                                    # every combination of these is solved
alpha_deg = [0.0, 10.0, 20.0]
beta_deg = [0.0, 5.0]
pairs = [[10.0, -4.0]]                    # optional: extra (alpha, beta) conditions
```

Everything for a case goes in one folder named after it, with inputs and results in files inside rather than in folder
names:

```
hpc/runs/<case name>/
    case.toml                    the case file as last submitted
    mesh/                        slurm_<jobid>.out, log.*, mesh_case.json, mesh_report.json, postProcessing/
    mesh.previous/               the last failed or unfinished mesh attempt, if any
    solve/<condition>/           e.g. solve/a+10.0_b+05.0/: slurm_<jobid>.out, log.*, solve_case.json,
                                 results.json, postProcessing/, final fields
    solve/<condition>.previous/  the last failed or unfinished attempt at that condition, if any
```

### Submission
`hpc/submit_cases.sh` submits a mesh job per case, then a solve job per flow condition that waits for the mesh (and is
cancelled if the mesh fails). It skips anything already finished or queued, so resubmitting a case only runs what is
missing. Flow conditions can be added to a meshed case and resubmitted; changing a meshed case's geometry, mesh
settings, deflections or overrides is refused instead of silently replacing the mesh (delete its `mesh/` folder to
remesh). The mesh is identified by those file paths, not their contents: after editing `configs/mesh/uav.toml`, delete
the affected `mesh/` folders to remesh.

| Option | Effect |
|---|---|
| `--dry-run` | Print the `sbatch` commands without submitting |
| `--mesh-only` | Submit only the mesh jobs |
| `--mesh-time`, `--solve-time HH:MM:SS` | Time limits (defaults 2 h and 6 h) |
| `--mesh-cores`, `--solve-cores N` | MPI ranks per job, up to 192 (one node); default 32, memory scaled to match |
| (default) | `sbatch --exclusive=user`: jobs share nodes only with your own jobs. Steady speed, but the first job on each node waits for one with no other users on it |
| `--shared` | Run on any free cores alongside other users' jobs: starts soonest, but OpenFOAM can be slowed 5–20× by neighbours competing for memory bandwidth |
| `--whole-node` | A node to itself per job |

The QOS allows 2 nodes per user at a time, so at 32 cores per job up to 12 jobs can run at once when Slurm packs them
onto the same nodes.

### Studies and batches
- **Mesh-independence study** (`configs/cases/mesh_study/`): three undeflected cases that differ only in
  `base_cell_size` (0.512, 0.384, 0.288 m, so every cell size changes by 4/3), each solved at α = 0, 10, 20, 30° and
  β = 0, 5°. The coarsest mesh whose coefficients agree with the next finer one (to within about 1–2%) is used for the
  dataset. The three-mesh set suits a Grid Convergence Index (Celik et al. 2008).
- **Vortex study** (`configs/cases/vortex_study/`): the medium mesh at α = 20°, β = 0 with five increasing
  `surface_distance_levels` refinements of the region above the skin, to see whether resolving the leading-edge
  shear layer and vortex reproduces the fine mesh's higher C_L.
- **Batch 2, actuator authority** (`configs/cases/batch2/`, 36 meshes, 233 solves, from
  `src/ml/taps/cfd_cases/cfd_cases2.csv`): a clean-wing map over α and β; symmetric deflection sweeps at
  α = −3, 3, 9, 15, 21, 27° (pitch and lift authority); one-sided (right-side) sweeps at α = 9, 21° and β = −10, 0, 10°
  (roll and yaw authority, the left side following by mirroring); and one combined full-deflection case. The upper and
  lower splitters always move together, up to 60°. Positive α only is run, with mostly the lift-reducing deflection
  direction, using the wing's top-bottom symmetry about α = −3°. Keeps the full 3D fields as well as the slices.
- **Batch 1** (`configs/cases/batch1/`, 121 meshes, 156 solves): an α sweep, a β sweep and a Latin hypercube over the
  12 actuators, α and β, from `src/ml/taps/latin_hyper.py`. Keeps only the slices and wall surfaces
  (`keep_fields = false`, written with `batch_cases.py --no-fields`). Its splitters still move independently.

The mesh study cases pin `surface_distance_levels = []` to stay as they were run. They were meshed before that
override existed, so resubmitting them to add conditions is refused; delete their `mesh/` folders first.

`src/cfd/batch_cases.py` turns a table of cases into case files: rows with the same deflections share one case file
(one mesh) with their (α, β) `pairs`, and `index.csv` maps every row to its case file and condition folder. The
deflection columns are `<surface>_deg` (the names in `geometry.toml`); older tables with `actuator_<k>_deg` columns are
read in `geometry.toml` order. Every row is checked against the deflection limits and the mesh's angle range before
anything is written.

## 6. Post-processing
Download the run folders: each `solve/<condition>/` needs `results.json`, `solve_case.json` and `postProcessing/`
(including `wallSurfaces/` for the pressure and y⁺ plots); the numbered field folders aren't needed. Then, from the
repository root:

```
.venv/Scripts/python.exe -m postpro.summary <run folders...>
.venv/Scripts/python.exe -m postpro.convergence <run folders...>
.venv/Scripts/python.exe -m postpro.pressure <run folders...> [--conditions a+10.0_b+00.0] [--stations 60 120 ...]
.venv/Scripts/python.exe -m postpro.boundary_layer <run folders...> [--conditions a+00.0_b+00.0]
```

Each takes run folders, a folder of run folders (a whole batch) or a glob pattern, writes tables and plots to
`<first path>/postpro/<script>/` (or `--out`), and prints the main tables; `--show` opens the figures. For a mesh
comparison, list the meshes coarse to fine: `pressure.py` compares each run with the last one. `summary.py` measures
control effects against the only undeflected case among the runs, or the one named with `--clean`.

## 7. Dataset (planned)
- **Mirror symmetry:** a case with deflections (left, right) at sideslip β mirrors the case (right, left) at −β, with
  C_Y, C_l, C_n changing sign and left and right surfaces swapping, so every solved case gives two dataset rows.
- **Collection:** the `results.json` files are collected into a Polars DataFrame, one row per case and condition
  (deflections, α, β, coefficients, hinge moments, convergence), saved as Parquet for `src/data/loaders.py`.
