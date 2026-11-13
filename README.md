# Reinforcement Learning Gust Control

A student research project on online estimation and reinforcement-learning control of unsteady aerodynamics in
large-amplitude gusts. Disturbances are detected from surface pressure taps on a 3D delta wing model, the unsteady
forces are estimated, and controllers hold a desired aerodynamic state (for example zero pitching moment) through gust
encounters. The modelling combines physics-augmented autoencoders (PA-AE), LSTM networks and reinforcement learning
(SAC).

## Code structure

| Folder | Contents |
|---|---|
| `src/ml/` | Learning: SAC (`algos/`), the policy, environments (`envs/`: dummy gust, XFOIL pitch, wind tunnel), autoencoder (`ae/`), distributed training, pressure-tap selection and CFD case design (`taps/`); train with `python -m src.ml.train <config>` |
| `src/cfd/` | CFD pipeline: geometry, control surface deflection, meshing (snappyHexMesh) and solving (simpleFoam) on Iridis 6 |
| `src/hw/` | Wind tunnel rig interface (DAQ, servos, safety limits) and a mock for running without the rig |
| `src/data/` | Data loading and checks (Polars) |
| `src/ui/` | Qt interface (`main.py`): displays results and launches the functional code, with no logic of its own |
| `ruststuff/` | Rust backend for performance-critical code, built into a Python module with maturin (stubs in `ruststuff.pyi`) |
| `postpro/` | Post-processing of downloaded CFD runs: summary tables, convergence, surface pressure, boundary layer |
| `hpc/` | Slurm job scripts and submission for Iridis 6 |
| `configs/` | Run configs (`sac_*.toml`), CFD mesh and solver settings, and CFD case files |
| `cad/` | Aircraft geometry: one STL per body and `geometry.toml` |
| `benchmarks/` | Benchmarks |

Python ≥ 3.14, managed with uv. Libraries: Polars (not Pandas), PyTorch, Pydantic, qtpy (UI), Matplotlib (one-off
plots) and the `ruststuff` backend. All code is fully type hinted.

## CFD

The controller is trained on a surrogate of the UAV's aerodynamics, built from steady RANS solutions over a range of
flow angles and control surface deflections. The pipeline runs end to end on Iridis 6:

```
CAD (one STL per body) ─► deflect surfaces, merge into one skin ─► snappyHexMesh, one mesh per set of deflections
    ─► simpleFoam k-ω SST, one job per (α, β) ─► results.json ─► postpro/ ─► (planned) dataset ─► surrogate
```

A case file (`configs/cases/`) names the deflections and flow conditions. `bash hpc/submit_cases.sh <case or folder>`
queues the mesh and solve jobs, and each solve reports the six aerodynamic coefficients, every control surface's hinge
moment, wall pressure and convergence. The `postpro/` scripts turn downloaded runs into tables and plots.

Current work: the mesh-independence study, a vortex-resolution study at α = 20°, and the actuator authority batch
(`configs/cases/batch2`).

Details (geometry rules, meshing and solver settings, case files, submission options, studies and post-processing):
[docs/cfd.md](docs/cfd.md).
