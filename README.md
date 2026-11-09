# Reinforcement Learning Gust Control — Project Overview

Project summary
---------------
This student research project develops online estimation and reinforcement-learning-based control methods for unsteady aerodynamic systems exposed to large-amplitude gusts. The aim is to detect aerodynamic disturbances from surface pressure sensors on a 2D wing, estimate the unsteady forces, and synthesise controllers that maintain desired aerodynamic performance (for example, constant lift) during gust encounters.

Modelling Approach
----------------
This project will aim to encorperate both Physics Augmented Auto Encoders (PA-AE), LSTM networks and RL to control the wing in extreme aerodynamic conditions. 

Basic Code Structure
---------------
This project includes a rust backend which simply compiles into a python module to allow us to right performative backend code whilst maintining the advantage of flexible pythoin libraries. 

In this project the following libraries will be used
- Polars (not Pandas)
- qtpy (for any custom UI)
- PyTorch (Faster, less overhead, iterable)
- ruststuff (Custom rust backend)
- Matplotlib (One off plotting)
- Pydantic (Serialisable and safe typing)

All code must simply be fully type hinted. 

CFD Process
---------------
The control model is trained on a surrogate of the delta wing UAV's aerodynamics, built from a dataset of RANS CFD solutions over a range of flow angles and control surface deflections. The pipeline is scripted end to end and runs on the Iridis 6 HPC: case files go in, and coefficients, hinge moments and surface pressures come out.

```
SolidWorks ─► one STL per body ─► deflect control surfaces ─► merge into one outer skin, check it
    ─► snappyHexMesh (once per set of deflections) ─► simpleFoam, k-ω SST (once per α, β) ─► results.json
    ─► (planned) Parquet dataset ─► surrogate
```

Status: geometry, meshing and the solver setup are implemented. Meshing has been run and checked on Iridis. The solver stage is written and checked locally but has not yet run on Iridis. Collecting results into a dataset is planned.

| Module | Purpose |
|---|---|
| `src/cfd/geometry.py` | Loads and validates `cad/<model>/geometry.toml`: bodies, hinges, deflection limits, reference values |
| `src/cfd/stl.py` | Reads ASCII and binary STL; writes multi-solid ASCII STL and OBJ edge files |
| `src/cfd/surface.py` | Welds triangles into a mesh, counts open edges, fills small holes |
| `src/cfd/deflect.py` | Loads the bodies and rotates control surfaces about their hinges |
| `src/cfd/skin.py` | Merges the bodies into one closed outer skin (boolean union), picks its feature edges, checks it; also a command to write and check a skin |
| `src/cfd/case.py` | Loads a case file: geometry, mesh and solver settings, deflections, flow conditions |
| `src/cfd/foam.py` | Helpers for writing OpenFOAM input files |
| `src/cfd/mesh.py` | Writes a ready-to-mesh OpenFOAM case (skin STL, feature edges, every dictionary) |
| `src/cfd/mesh_report.py` | Reads the checkMesh and snappyHexMesh logs, applies the acceptance limits, writes `mesh_report.json` |
| `src/cfd/solve.py` | Writes a ready-to-run simpleFoam case for one flow condition on a finished mesh |
| `src/cfd/solve_report.py` | Turns the solver output into coefficients, hinge moments and a convergence verdict (`results.json`) |
| `src/cfd/submit.py` | Submits mesh and solve jobs to Slurm, skipping finished work (run through `hpc/submit_cases.sh`) |
| `src/cfd/runs.py` | Lists every run and its state, read from the run folders |
| `src/cfd/plot_solves.py` | Plots convergence histories and coefficients against α for one or more downloaded run folders, and prints the slopes |

| HPC file | Purpose |
|---|---|
| `hpc/setup_python.sh` | One-off: builds `.venv-hpc` (CPU PyTorch, Pydantic, manifold3d) on the login node |
| `hpc/submit_cases.sh` | Submits case files (wrapper around `src/cfd/submit.py`) |
| `hpc/mesh_case.slurm` | Mesh job: writes the case, blockMesh, snappyHexMesh, checkMesh, report |
| `hpc/solve_case.slurm` | Solve job: writes the case, potentialFoam, simpleFoam, report |
| `hpc/env.sh` | Shared setup for the above: loads OpenFOAM v2506 and the Python environment |

### Quick start
1. **Geometry** (only when the CAD changes): export each body as an STL into `cad/uav/` and check the file names in `cad/uav/geometry.toml` (see section 1).
2. **Case file**: write `configs/cases/<name>.toml` (see section 5). To check the geometry first (about 20 s):
   `uv run python -m src.cfd.skin cad/uav/geometry.toml check/skin.stl --cad-units -d wingtip_r=10 wingtip_l=10`
3. **On Iridis**: `git pull`, and after a dependency change `bash hpc/setup_python.sh`.
4. **Submit**: `bash hpc/submit_cases.sh configs/cases/<name>.toml`, or a folder of case files. Add `--dry-run` to see what would be submitted, `--mesh-only` to stop before solving, and `--mesh-time` / `--solve-time HH:MM:SS` to change the time limits.
5. **Monitor**: `squeue -u $USER` lists jobs as `mesh_<case>` and `solve_<case>_<condition>`. `tail -f hpc/runs/<case>/mesh/slurm_<jobid>.out` follows a job.
6. **Check**: after `module load $(cat .venv-hpc/python_module) && source .venv-hpc/bin/activate`, run `python -m src.cfd.runs` for one line per case, or `python -m src.cfd.runs --solves` for one line per flow condition with its coefficients.
7. **Plot** (locally): download the run folders, then run `uv run python -m src.cfd.plot_solves <data folder>/<case> ...` in the VS Code terminal. It writes convergence plots per case and one coefficient figure (all six coefficients against α and the drag polar, every case overlaid) to `<first run folder>/plots/`. It also prints the coefficients, the lift-curve and pitching-moment slopes with the static margin, and the sideslip derivatives. `--show` opens the figures.

### 1. Geometry
- The aircraft is a SolidWorks multibody part. The CFD uses only the outer (wetted) surface. The main body and each control surface are separate bodies: the current model (`cad/uav/`) has a main body and six control surfaces per side (wingtip, leading-edge flap, outboard and inboard trailing-edge flaps, upper and lower splitter flaps).
- Each body is exported as its own STL, at zero deflection, in the SolidWorks part frame (x aft, y starboard, z up) in mm, and becomes its own OpenFOAM patch, so forces and hinge moments can be read per surface.
- `cad/<model>/geometry.toml` lists every body with its STL file, each control surface's hinge axis and deflection limits, and the reference point, chord, span and area used for the coefficients. Unknown keys, bad names and inconsistent limits are errors on load.

**STL export settings** (Save As → STL → Options): binary, mm, **Custom** resolution with deviation 0.04 mm (the smallest SolidWorks allows) and angle 1°; tick **"Do not translate STL output data to positive space"** (otherwise the hinge points no longer line up; `stl_offset` in `geometry.toml` can correct a shifted export); untick **"Save all components of an assembly in a single file"**. Check the file names afterwards: SolidWorks sometimes changes the `-3` / `-4` suffix.

**How bodies may meet.** The bodies are merged into one outer skin before meshing, which only gives a clean skin if they **overlap, or meet on flat faces in a plane of constant x, y or z**. Avoid:
- a curved surface shared by two bodies with zero clearance, such as a flap nose in a cove of the same radius. The exports are triangulated separately and weave through each other, leaving slivers. Fill the cove so the nose is buried in the main body (done for every flap).
- flush faces at an angle to the axes, such as the swept leading-edge flap ends. STL stores single-precision coordinates, so the two copies end up a hair apart. Make the bodies overlap (the leading-edge flaps are 0.3 mm longer at each end).
- slits and near-misses, such as a loft ending just short of the wing. Extend one body into the other.

Every body must be closed (watertight); the current exports are. Small holes up to `max_hole_perimeter` (20 mm) are filled on load as a safety net, and a larger one stops the pipeline with its location.

The left and right exports are triangulated slightly differently by SolidWorks, so the two sides of a symmetric mesh can differ in small details (for example around the wingtip housings).

### 2. Control surfaces and the outer skin
Each hinge axis is given by two points, `p1` and `p2`. Positive deflection is a right-hand rotation about the axis from `p1` to `p2` (Rodrigues' formula: `p' = a + (p−a)cosδ + (k × (p−a))sinδ + k(k·(p−a))(1−cosδ)`). Port surfaces are mirror images of the starboard ones, and a mirror reverses the rotation sense, so each port hinge uses the mirrored points in swapped order; the same positive angle then gives the same physical motion on both sides.

| Surface | Positive deflection |
|---|---|
| Wingtip, inboard and outboard trailing-edge flaps | trailing edge up |
| Leading-edge flap | leading edge down |
| Upper and lower splitter flaps (0 to 90°) | open (upper swings up, lower swings down) |

For each case the deflected bodies are merged into one closed surface, their boolean union (the `manifold3d` library), which cuts the triangles where bodies meet and keeps only the outside, so no hidden faces are left for snappyHexMesh to refine around. Each triangle remembers which body it came from, and the skin is written as `aircraft.stl` with one named solid (patch) per body. The skin is then checked: it must be closed, and no part of it may face another across a gap narrower than 0.1 mm. A failure names the bodies and the location; `python -m src.cfd.skin` writes the skin, its feature edges (`<name>_features.obj`) and any offending faces (`<name>_narrow_gaps.stl`) for viewing in ParaView.

**Feature edges** (where snappyHexMesh refines and snaps to sharp edges) are chosen in `skin.py`, not by `surfaceFeatureExtract`. Where two bodies meet, their separate triangulations leave a step of a few hundredths of a millimetre along the seam, which a plain angle test sees as a sharp edge, refining every seam and denting the surface. An edge counts only if it turns by more than 30° and both of its faces come from source triangles that are mostly on the outside, which every seam step fails.

### 3. Meshing (snappyHexMesh)
- One mesh per set of deflections, reused for every flow condition. Settings are in `configs/mesh/uav.toml` (its comments give the cell size of each level); a case can change individual values with `[mesh_overrides]`.
- Background box about 20 aircraft lengths upstream, above, below and to each side, and 40 downstream, centred on the centreline with a cell face on y = 0, so a symmetric aircraft is meshed the same on both sides.
- Uniform 1.5 mm surface cells, refined to 0.75 mm at sharp curvature (leading edges) and along feature edges; 12 mm cells in a box around the aircraft and in the wake; 5 prism layers on every wall.
- The six outer faces form one `freestream` patch, and the wake box is widened to contain the wake over the whole angle range in the mesh settings (`min_alpha_deg`, `max_alpha_deg`, `max_beta_deg`: currently α from −10° to 30°, β up to ±10°). So one mesh serves every flow direction in that range, and flow conditions outside it are rejected.
- The job finishes with `checkMesh` and `mesh_report.json`: cell counts, maximum non-orthogonality (limit 70°) and skewness (limit 4), and prism-layer coverage per wall (warning below 80%). The job fails if the mesh fails. `postProcessing/meshSlices/` holds cutting planes and the wall surfaces with quality fields, and the walls also carry the layer coverage (`nSurfaceLayers`, `thicknessFraction`); all small enough to download and open in ParaView. Faces that checkMesh flags as bad are written as VTK under `postProcessing/constant/`.
- The medium mesh (`base_cell_size` 0.384 m) has about 2.7M cells before the widened wake box and takes about 3 minutes on 32 cores (one run on a slow node took 30).

### 4. Solving (simpleFoam)
- Steady, incompressible RANS with the k-ω SST turbulence model; settings in `configs/solver/uav_rans.toml` (airspeed, air properties, freestream turbulence, iterations, convergence). **The airspeed is a placeholder (20 m/s) until the design cruise speed is set.**
- Walls use Spalding's wall function (`nutUSpaldingWallFunction`), valid across the y⁺ ≈ 10–30 that the prism layers give. The `freestream` patch uses `freestreamVelocity` / `freestreamPressure`, which act as inflow or outflow wherever the flow enters or leaves.
- Flow direction in the model frame (x aft, y starboard, z up): **U = V (cos α cos β, −sin β, sin α cos β)**. Positive α means the air comes from below the nose, positive β from the starboard side. Lift is measured perpendicular to U in the x-z plane, drag along U.
- Each condition starts from a potential-flow solution (`potentialFoam`) and runs simpleFoam up to `max_iterations`, stopping early once C_L and C_D settle. Conditions are independent, so they run as separate jobs in parallel.
- **Results** (`results.json`, means over the last 200 iterations with their ranges): C_L, C_D, C_Y; C_l, C_m, C_n about the reference point in the usual flight-mechanics sense (right wing down, nose up, nose right positive); the hinge moment of each control surface about its own hinge axis (positive in the direction of positive deflection), in N·m and as C_h = M_h / (q S c); forces and moments in newtons; final residuals. A condition is converged if C_L, C_D and C_m each vary by less than 0.1% over that window. An unconverged condition (for example unsteady separated flow at high α) is still reported, with its ranges showing how much it oscillated.
- Also written: the force, moment and residual histories in `postProcessing/`, and wall pressure, wall shear stress and y⁺ as VTK (`postProcessing/wallSurfaces/`). The final 3D fields (about 200 MB per condition) are kept unless `keep_fields = false`.

### 5. Case files, run folders and submission
A case file in `configs/cases/` (subfolders allowed) describes one set of deflections and the flow conditions to solve it at. Its file name is the case name, which must be unique:

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
[flow]                                    # every combination of these is solved
alpha_deg = [0.0, 10.0, 20.0]
beta_deg = [0.0, 5.0]
```

Everything for a case goes in one folder named after it, with inputs and results in files inside rather than in folder names:

```
hpc/runs/<case name>/
    case.toml                    the case file as last submitted
    mesh/                        slurm_<jobid>.out, log.*, mesh_case.json, mesh_report.json, postProcessing/
    mesh.previous/               the last failed or unfinished mesh attempt, if any
    solve/<condition>/           e.g. solve/a+10.0_b+05.0/: slurm_<jobid>.out, log.*, solve_case.json,
                                 results.json, postProcessing/, final fields
```

`hpc/submit_cases.sh` submits a mesh job per case, then a solve job per flow condition that waits for the mesh (and is cancelled if the mesh fails). It skips anything already finished or queued, so resubmitting a case only runs what is missing. Flow conditions can be added to a meshed case and resubmitted; changing a meshed case's geometry, mesh settings, deflections or overrides is refused instead of silently replacing the mesh (delete its `mesh/` folder to remesh). The mesh is identified by those file paths, not their contents: after editing `configs/mesh/uav.toml`, delete the affected `mesh/` folders to remesh.

**Mesh-independence study:** `configs/cases/mesh_study/` has three undeflected cases that differ only in `base_cell_size` (0.512, 0.384, 0.288 m, so every cell size changes by 4/3), each solved at α = 0, 10, 20, 30° and β = 0, 5°. The coarsest mesh whose coefficients agree with the next finer one (to within about 1–2%) is used for the dataset. Submit the meshes with `bash hpc/submit_cases.sh configs/cases/mesh_study --mesh-only`, then, once a single test solve has worked, the solves with `bash hpc/submit_cases.sh configs/cases/mesh_study --solve-time 08:00:00` (finished meshes are skipped).

**Layer tests:** `configs/cases/mesh_tests/` holds mesh-only variants of the fine study mesh, each changing one layer setting (`n_layer_iter`, `layer_shrinker`, `n_grow`), to compare layer coverage and checkMesh results with `mesh_fine`. The setting that works best goes into `configs/mesh/uav.toml`, so all three study meshes use it.

### 6. Dataset (planned)
- **Sampling:** deflections sampled with a space-filling design (Latin hypercube or Sobol), each case solved over a grid of α and β.
- **Mirror symmetry:** a case with deflections (left, right) at sideslip β mirrors the case (right, left) at −β, with C_Y, C_l, C_n changing sign and left and right surfaces swapping, so every solved case gives two dataset rows.
- **Collection:** the `results.json` files are collected into a Polars DataFrame, one row per case and condition (deflections, α, β, coefficients, hinge moments, convergence), saved as Parquet for `src/data/loaders.py`.
