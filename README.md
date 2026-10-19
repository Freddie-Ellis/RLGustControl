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
The control model is trained on a surrogate model of the delta wing UAV's aerodynamics. The surrogate is built from a dataset of RANS CFD solutions covering a range of flight conditions and control surface deflections. The pipeline is scripted end to end: a geometry and a list of cases go in, and a table of forces, moments and surface pressures comes out.

```
SolidWorks part ─► one STL per body ─► fill small holes ─► rotate control surfaces about hinge axes (per case)
        ─► snappyHexMesh ─► simpleFoam (RANS, k-ω SST) ─► post-process ─► Parquet dataset ─► surrogate
```

Status: steps 1 and 2 are implemented in `src/cfd/`, and step 3 has a first test version. Steps 4 and 5 are planned.

| Module | Purpose |
|---|---|
| `src/cfd/geometry.py` | Loads and validates `cad/<model>/geometry.toml` (bodies, hinges, limits, reference values) |
| `src/cfd/stl.py` | Reads ASCII and binary STL files; writes multi-solid ASCII STL |
| `src/cfd/surface.py` | Checks that a surface is closed and fills small holes |
| `src/cfd/deflect.py` | Rotates control surfaces and builds the deflected aircraft; also runs as a script |
| `src/cfd/case.py` | Loads a case file (`configs/cases/*.toml`): geometry, mesh settings and deflections for one case |
| `src/cfd/mesh.py` | Writes a ready-to-mesh OpenFOAM case (aircraft STL and every dictionary) from a case file |
| `src/cfd/mesh_report.py` | Reads the checkMesh and snappyHexMesh logs, applies the acceptance limits, writes `mesh_report.json` |

### 1. Geometry
- The aircraft is modelled in SolidWorks as a multibody part. The CFD uses only the outer (wetted) surface. Internal structure and small features such as fasteners and antennas are left out.
- The main body and each control surface are separate solid bodies. The current model (`cad/uav/`) has a main body and six control surfaces per side: wingtip, leading-edge flap, outboard and inboard trailing-edge flaps, and upper and lower splitter flaps.
- Each body is exported as its own STL, at zero deflection, in the SolidWorks part frame (x aft, y starboard, z up) and in mm. Each body becomes a separate OpenFOAM patch, so forces and hinge moments can be read per control surface.
- `cad/<model>/geometry.toml` lists every body with its STL file, and for each control surface its hinge axis and deflection limits. It also holds the reference point, length and area used to normalise the force coefficients. Unknown keys, bad names and inconsistent limits are reported as errors on load.

**SolidWorks STL export settings** (Save As → STL → Options):
- Binary output, units mm, **Custom** resolution: deviation about 0.01–0.02 mm and angle about 1–2°. Coarser settings leave the leading edges visibly faceted, and snappyHexMesh copies that faceting into the mesh.
- Tick **"Do not translate STL output data to positive space"**. Otherwise SolidWorks shifts the model and the hinge points no longer line up with it. If an export has been shifted, set `stl_offset` in `geometry.toml` to the shift.
- Untick **"Save all components of an assembly in a single file"**, so you get one file per body.

**Closed surfaces:** every body must be closed (watertight) to mesh reliably. SolidWorks leaves thin slits where the loft meets the wing on the main body, even after healing the part as a Parasolid. These are filled automatically on load, as long as each hole's perimeter is at most `max_hole_perimeter` in `geometry.toml` (currently 20 mm; the largest slit is 13.1 mm). A bigger hole stops the pipeline with its size and location, so a real gap in the CAD can't be covered over unnoticed.

### 2. Control surface deflection
Each hinge axis is given as two points, `p1` and `p2`, measured in SolidWorks. Positive deflection is a right-hand rotation about the axis pointing from `p1` to `p2`: point your right thumb from `p1` to `p2`, and your fingers curl in the positive direction. Swapping the points reverses the sign.

The port surfaces are mirror images of the starboard ones. A mirror reverses the sense of rotation, so each port hinge uses the mirrored points in swapped order. That way the same positive angle gives the same physical motion on both sides. Current sign conventions:

| Surface | Positive deflection |
|---|---|
| Wingtip, inboard and outboard trailing-edge flaps | trailing edge up |
| Leading-edge flap | leading edge down |
| Upper and lower splitter flaps | open (upper swings up, lower swings down) |

For each case, every control surface is rotated about its hinge by its deflection (surfaces not listed stay at 0) and converted to metres. The whole aircraft is written as one ASCII STL, `aircraft.stl`, with each body as its own named solid. Each solid becomes its own OpenFOAM patch, named after the body. Bodies are not merged into one surface: where a deflected surface overlaps a fixed one, snappyHexMesh keeps only the fluid outside both. Angles outside a surface's limits are rejected.

Rotation of a point **p** about the hinge axis through **a** with unit direction **k** by angle δ (Rodrigues' formula):

```
p' = a + (p−a)cosδ + (k × (p−a))sinδ + k(k·(p−a))(1−cosδ)
```

To write the deflected aircraft and check it in a viewer (`--cad-units` writes mm in the part frame, so it overlays on the SolidWorks model):

```
uv run python -m src.cfd.deflect cad/uav/geometry.toml hpc/runs/check/aircraft.stl -d wingtip_r=10 wingtip_l=10 --cad-units
```

In Python, `write_case_geometry(geometry, {"wingtip_r": 10.0}, path)` writes the aircraft STL for one case in metres.

### 3. Meshing (snappyHexMesh)
- `blockMesh` builds a background hex domain extending about 20 body lengths upstream and to the sides, and about 40 lengths downstream.
- `snappyHexMesh` snaps to the named surfaces, with:
  - surface refinement on the leading edges and the control surfaces, including where they meet the main body;
  - region refinement over the upper surface and into the wake, where the leading edge vortices are;
  - prism layers on all walls.
- The full aircraft is meshed (no symmetry plane), because asymmetric control surface deflections and sideslip make the flow asymmetric.
- Angle of attack α and sideslip β are set through the inlet velocity direction and the `forceCoeffs` lift/drag directions, so one mesh covers every α and β for a given set of deflections.
- `checkMesh` runs on every case, and cases that fail the quality limits are flagged.

**Case files:** each case is a TOML file in `configs/cases/`, and the file name is the case name. It names the geometry, the mesh settings and the deflections, so a job needs nothing else:

```toml
description = "Both wingtips at +10 deg"
geometry = "cad/uav/geometry.toml"
mesh = "configs/mesh/uav_test.toml"
[deflections_deg]          # surfaces not listed stay at 0
wingtip_r = 10.0
wingtip_l = 10.0
```

**Running a mesh on Iridis 6** (the whole process runs on the HPC):
1. Once, on the login node: `bash hpc/setup_python.sh`. It creates `.venv-hpc` with the CPU build of PyTorch and Pydantic.
2. Submit one case: `sbatch hpc/mesh_case.slurm configs/cases/uav_wingtip_10.toml`. To submit many, put their case files in one folder and run `bash hpc/submit_cases.sh <folder>`. This submits a Slurm array with one task per case file.
3. The job checks the case file, then writes the case (aircraft STL and dictionaries) to `hpc/runs/<case name>_<jobid>/`. It then runs `surfaceFeatureExtract`, `blockMesh`, `snappyHexMesh` (MPI), `reconstructParMesh` and `checkMesh -allGeometry -allTopology`.
4. It finishes with `mesh_report.json`: cell counts and types, maximum non-orthogonality and skewness, prism-layer coverage per wall, and pass/fail against the `[acceptance]` limits in the mesh settings. The job exits with an error if the mesh fails.
5. To inspect the mesh, download `postProcessing/meshSlices/`. It holds cutting planes through the root and each starboard control surface, plus the wall surfaces, as small VTK files carrying cell quality fields. Faces that checkMesh flags as bad are in `postProcessing/checkMesh/`. The full mesh is too large to open on a laptop.

Mesh settings (domain size, refinement levels, prism layers, acceptance limits) are in `configs/mesh/uav_test.toml`; the comments there give the cell size for each level.

### 4. Solving (OpenFOAM on the HPC)
- **Solver:** `simpleFoam` (steady, incompressible RANS).
- **Turbulence model:** k-ω SST.
- **Outputs** (function objects):
  - `forceCoeffs` for the whole aircraft and for each control surface patch (C_L, C_D, C_Y, C_l, C_m, C_n, and hinge moments);
  - surface pressure sampled at the pressure sensor locations;
  - residual histories.
- **Convergence:** residuals below 1e-5, and force coefficients changing by less than 0.1 % over the final iterations.
- **HPC:** OpenFOAM runs on the Linux HPC. Each case is one element of a Slurm array job and runs in parallel with `decomposePar` and MPI.

### 5. Case generation and the dataset
- **Sampling:** α, β, airspeed and each control surface deflection are sampled with a space-filling design (Latin hypercube or Sobol).
- **Case generator (Python):** reads the case config, copies a template OpenFOAM case, writes the deflected STLs and boundary conditions, and creates the Slurm array script.
- **Post-processing:** the function object outputs from all cases are collected into a Polars DataFrame, with one row per case (inputs, coefficients, sensor pressures and convergence flags), and saved as Parquet for `src/data/loaders.py`.
