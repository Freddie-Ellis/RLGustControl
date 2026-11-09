"""Build an OpenFOAM case that solves one flow condition on a finished mesh with steady RANS.

A case (one set of deflections) is meshed once and solved at every flow condition in its [flow] table, each in
its own folder hpc/runs/<case name>/solve/<condition>/, e.g. solve/a+08.0_b+00.0/. `write_solve_case` writes:

    constant/polyMesh                  link to the case's mesh (hpc/runs/<case name>/mesh/constant/polyMesh)
    constant/transportProperties       air viscosity
    constant/turbulenceProperties      k-omega SST
    0/U, p, k, omega, nut              initial and boundary conditions
    system/controlDict                 iterations, and the function objects that record the results
    system/fvSchemes, fvSolution       discretisation and linear solvers
    system/decomposeParDict            parallel decomposition
    solve_case.json                    every setting used, the flow condition, the job id

The physics:
    - simpleFoam: steady, incompressible RANS (the UAV flies far below Mach 0.3), k-omega SST turbulence model.
    - Walls use Spalding's wall function (nutUSpaldingWallFunction), valid whether the first cell lies in the
      viscous sublayer, the buffer layer or the log layer. The prism layers put it at y+ ~ 10-30.
    - The outer boundary is one `freestream` patch. freestreamVelocity / freestreamPressure act as an inlet
      wherever the flow enters and an outlet wherever it leaves, so the flow direction is set only by the
      freestream velocity vector and one mesh serves every angle of attack and sideslip.

Flow direction, in the model frame (x aft, y starboard, z up), for angle of attack alpha and sideslip beta:
    freestream velocity   U = V (cos(alpha) cos(beta), -sin(beta), sin(alpha) cos(beta))
At positive alpha the air comes from below the nose (U has an upward component); at positive beta it comes
from the starboard side (U points to port). Lift is perpendicular to U in the x-z plane, (-sin(alpha), 0,
cos(alpha)), and drag is along U.

Function objects (written to postProcessing/):
    forces_aircraft          total force and moment on all walls, moments about the reference point
    forces_<surface>         force and moment on each control surface, moments about a point on its hinge
                             (the hinge moment is the component along the hinge axis; see solve_report.py)
    forceCoeffs_aircraft     lift and drag coefficients for monitoring, and for stopping on convergence
    converged                stops the run once the lift and drag coefficients have settled
    solverInfo               residuals at every iteration
    yPlus, wallShearStress   written with the final fields
    wallSurfaces             pressure, wall shear stress and y+ on the walls as VTK, to download

Run as a script (normally from the Slurm job, hpc/solve_case.slurm):
    python -m src.cfd.solve conditions configs/cases/uav_clean.toml       list the case's flow conditions
    python -m src.cfd.solve write configs/cases/uav_clean.toml hpc/runs/uav_clean/solve/a+08.0_b+00.0 \\
        --alpha 8 --beta 0 --n-procs 32
"""

import argparse
import json
import math
import os
import re
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

from pydantic import Field

from src.cfd.case import CaseSpec, FlowCondition
from src.cfd.foam import foam_vector, write_foam_file
from src.cfd.geometry import GeometryConfig, StrictModel
from src.cfd.mesh import MeshSettings

FREESTREAM = "freestream"  # the outer boundary patch, written by mesh.py

type Vec3 = tuple[float, float, float]


class SolverSettings(StrictModel):
    """Solver settings, loaded from a TOML file (see configs/solver/). SI units."""
    speed: float = Field(default=20.0, gt=0)                  # freestream airspeed (m/s)
    density: float = Field(default=1.225, gt=0)               # air density (kg/m^3), turns forces into newtons
    kinematic_viscosity: float = Field(default=1.5e-5, gt=0)  # air (m^2/s)

    # Freestream turbulence: intensity (fluctuation / airspeed) and eddy-to-molecular viscosity ratio.
    turbulence_intensity: float = Field(default=0.005, gt=0)
    viscosity_ratio: float = Field(default=10.0, gt=0)

    # No residual stopping rule: started from potential flow, the residuals fell below 1e-5 within ~150
    # iterations while the forces were still changing. Runs stop on the force coefficients (below) or here.
    max_iterations: int = Field(default=3000, gt=0)

    # Converged when, over the last convergence_window iterations, each of C_L, C_D and C_m varies by less than
    # convergence_tolerance * max(|mean|, 0.1). The run stops early once C_L and C_D meet this (after
    # convergence_start iterations, so the starting transient can't trigger it).
    convergence_window: int = Field(default=200, gt=0)
    convergence_tolerance: float = Field(default=1e-3, gt=0)
    convergence_start: int = Field(default=500, ge=0)

    n_non_orthogonal_correctors: int = Field(default=1, ge=0)  # the meshes reach ~70 deg non-orthogonality
    keep_fields: bool = True  # keep the final 3D fields (~200 MB per condition); results are kept regardless

    @classmethod
    def from_toml(cls, path: Path) -> SolverSettings:
        with open(path, "rb") as f:
            return cls.model_validate(tomllib.load(f))

    def freestream_k(self) -> float:
        """Turbulent kinetic energy k = 3/2 (I V)^2."""
        return 1.5 * (self.turbulence_intensity * self.speed) ** 2

    def freestream_omega(self) -> float:
        """Specific dissipation omega = k / nu_t, with nu_t = viscosity_ratio * nu."""
        return self.freestream_k() / (self.viscosity_ratio * self.kinematic_viscosity)


def freestream_direction(condition: FlowCondition) -> Vec3:
    """Unit vector the freestream flows along, in the model frame (see the module docstring)."""
    a, b = math.radians(condition.alpha_deg), math.radians(condition.beta_deg)
    return (math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b))


def lift_direction(condition: FlowCondition) -> Vec3:
    """Unit vector lift is measured along: perpendicular to the freestream, in the x-z plane, upward."""
    a = math.radians(condition.alpha_deg)
    return (-math.sin(a), 0.0, math.cos(a))


def mesh_patches(mesh_dir: Path) -> list[str]:
    """Patch names in a mesh's polyMesh/boundary file: the names of the entries in its list, which starts
    after the FoamFile header with a line holding just "(" ."""
    text = (mesh_dir / "constant" / "polyMesh" / "boundary").read_text()
    start = re.search(r"^\s*\(\s*$", text, re.MULTILINE)
    if start is None:
        raise ValueError(f"{mesh_dir}: can't read the patch list in constant/polyMesh/boundary")
    return re.findall(r"^\s*([A-Za-z_]\w*)\s*\n\s*\{", text[start.end():], re.MULTILINE)


def check_condition(condition: FlowCondition, mesh: MeshSettings) -> None:
    """Raise if a flow condition lies outside the angle range the mesh's wake box was sized for."""
    if not mesh.min_alpha_deg <= condition.alpha_deg <= mesh.max_alpha_deg:
        raise ValueError(f"alpha {condition.alpha_deg} deg is outside the mesh's range "
                         f"[{mesh.min_alpha_deg}, {mesh.max_alpha_deg}] (min/max_alpha_deg in the mesh settings)")
    if abs(condition.beta_deg) > mesh.max_beta_deg:
        raise ValueError(f"beta {condition.beta_deg} deg is outside the mesh's range +-{mesh.max_beta_deg} "
                         "(max_beta_deg in the mesh settings)")


def _link_mesh(mesh_dir: Path, solve_dir: Path) -> None:
    """Point solve_dir/constant/polyMesh at the mesh, without copying it (a relative symbolic link), or copy
    it where links aren't allowed (e.g. Windows without developer mode)."""
    source = mesh_dir / "constant" / "polyMesh"
    target = solve_dir / "constant" / "polyMesh"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    try:
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def write_solve_case(
    geometry: GeometryConfig,
    settings: SolverSettings,
    condition: FlowCondition,
    mesh_dir: Path,
    solve_dir: Path,
    n_procs: int,
) -> dict[str, object]:
    """Write a complete, ready-to-run simpleFoam case for one flow condition on the mesh in `mesh_dir`.

    Returns the case summary that is also saved as solve_case.json.
    """
    s = settings
    patches = mesh_patches(mesh_dir)
    if FREESTREAM not in patches:
        raise ValueError(f"{mesh_dir}: no '{FREESTREAM}' patch; remesh the case with the current mesh.py")
    walls = [p for p in patches if p != FREESTREAM]
    wall_regex = "(" + "|".join(walls) + ")"
    _link_mesh(mesh_dir, solve_dir)

    scale = geometry.scale_to_m
    ref = geometry.reference
    ref_point = tuple(x * scale for x in ref.point)
    u_inf = tuple(s.speed * c for c in freestream_direction(condition))
    k_inf, omega_inf = s.freestream_k(), s.freestream_omega()

    # Initial and boundary conditions. Every field starts uniform at its freestream value.
    def field(name: str, foam_class: str, dimensions: str, internal: str, freestream: str, wall: str) -> None:
        write_foam_file(solve_dir, f"0/{name}", f"""
dimensions      {dimensions};
internalField   {internal};

boundaryField
{{
    {FREESTREAM}
    {{
{freestream}
    }}
    "{wall_regex}"
    {{
{wall}
    }}
}}
""", foam_class)

    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", f"uniform {foam_vector(u_inf)}",
          f"        type freestreamVelocity;\n        freestreamValue uniform {foam_vector(u_inf)};",
          "        type noSlip;")
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0",
          "        type freestreamPressure;\n        freestreamValue uniform 0;",
          "        type zeroGradient;")
    field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k_inf:.6g}",
          f"        type inletOutlet;\n        inletValue uniform {k_inf:.6g};\n        value uniform {k_inf:.6g};",
          f"        type kqRWallFunction;\n        value uniform {k_inf:.6g};")
    field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {omega_inf:.6g}",
          f"        type inletOutlet;\n        inletValue uniform {omega_inf:.6g};\n"
          f"        value uniform {omega_inf:.6g};",
          f"        type omegaWallFunction;\n        value uniform {omega_inf:.6g};")
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0",
          "        type calculated;\n        value uniform 0;",
          "        type nutUSpaldingWallFunction;\n        value uniform 0;")

    write_foam_file(solve_dir, "constant/transportProperties", f"""
transportModel  Newtonian;
nu              {s.kinematic_viscosity:.6g};
""")
    write_foam_file(solve_dir, "constant/turbulenceProperties", """
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""")

    # One forces object per control surface, with moments about a point on its hinge: the hinge doesn't move
    # when the surface deflects, so the moment's component along the hinge axis is the hinge moment.
    surface_forces = "\n".join(f"""
    forces_{c.name}
    {{
        type            forces;
        libs            (forces);
        patches         ({c.name});
        rho             rhoInf;
        rhoInf          {s.density:g};
        CofR            {foam_vector([x * scale for x in c.hinge.p1])};
        writeControl    timeStep;
        writeInterval   1;
    }}""" for c in geometry.control_surfaces if c.name in walls)

    write_foam_file(solve_dir, "system/controlDict", f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {s.max_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {s.max_iterations};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    forces_aircraft
    {{
        type            forces;
        libs            (forces);
        patches         ({" ".join(walls)});
        rho             rhoInf;
        rhoInf          {s.density:g};
        CofR            {foam_vector(ref_point)};
        writeControl    timeStep;
        writeInterval   1;
    }}
{surface_forces}

    forceCoeffs_aircraft
    {{
        type            forceCoeffs;
        libs            (forces);
        patches         ({" ".join(walls)});
        rho             rhoInf;
        rhoInf          {s.density:g};
        magUInf         {s.speed:g};
        lRef            {ref.length * scale:.6g};
        Aref            {ref.area * scale**2:.6g};
        CofR            {foam_vector(ref_point)};
        dragDir         {foam_vector(freestream_direction(condition))};
        liftDir         {foam_vector(lift_direction(condition))};
        writeControl    timeStep;
        writeInterval   1;
    }}

    // Stop once lift and drag have settled (see SolverSettings), writing the final fields.
    converged
    {{
        type            runTimeControl;
        libs            (utilityFunctionObjects);
        timeStart       {s.convergence_start};
        conditions
        {{
            coefficients
            {{
                type            average;
                functionObject  forceCoeffs_aircraft;
                fields          (Cl Cd);
                tolerance       {s.convergence_tolerance:g};
                window          {s.convergence_window};
                windowType      approximate;
            }}
        }}
    }}

    solverInfo
    {{
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (U p k omega);
        writeResidualFields no;
    }}

    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         ({" ".join(walls)});
        writeControl    writeTime;
    }}

    wallSurfaces
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   vtk;
        interpolationScheme cell;
        fields          (p wallShearStress yPlus);
        surfaces
        {{
            walls {{ type patch; patches ("{wall_regex}"); interpolate false; }}
        }}
    }}
}}
""")

    write_foam_file(solve_dir, "system/fvSchemes", """
ddtSchemes      { default steadyState; }

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

// Limited non-orthogonal correction: the meshes reach ~70 deg non-orthogonality.
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
""")

    write_foam_file(solve_dir, "system/fvSolution", f"""
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }}
    Phi
    {{
        $p;
    }}
    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors {s.n_non_orthogonal_correctors};
    consistent      yes;   // SIMPLEC: no pressure under-relaxation needed
}}

// potentialFoam, run first to give simpleFoam a better starting field than uniform flow.
potentialFlow
{{
    nNonOrthogonalCorrectors 10;
}}

relaxationFactors
{{
    equations
    {{
        U               0.9;
        k               0.7;
        omega           0.7;
    }}
}}
""")
    write_foam_file(solve_dir, "system/decomposeParDict", f"numberOfSubdomains {n_procs};\nmethod scotch;")

    info: dict[str, object] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "geometry": geometry.name,
        "condition": condition.model_dump(),
        "freestream_velocity_m_s": u_inf,
        "lift_direction": lift_direction(condition),
        "freestream_k": k_inf,
        "freestream_omega": omega_inf,
        "walls": walls,
        # What solve_report.py needs to turn forces into coefficients and hinge moments, in metres.
        "reference": {
            "point_m": ref_point,
            "length_m": ref.length * scale,
            "span_m": ref.span * scale,
            "area_m2": ref.area * scale**2,
        },
        "hinges": {
            c.name: {"point_m": [x * scale for x in c.hinge.p1], "direction": c.hinge.direction}
            for c in geometry.control_surfaces if c.name in walls
        },
        "n_procs": n_procs,
        "settings": settings.model_dump(),
    }
    (solve_dir / "solve_case.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write an OpenFOAM case that solves one flow condition.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("conditions", help="print the case's flow conditions: alpha beta name")
    listing.add_argument("case_file", type=Path)
    write = commands.add_parser("write", help="write the solve case for one flow condition")
    write.add_argument("case_file", type=Path)
    write.add_argument("solve_dir", type=Path)
    write.add_argument("--alpha", type=float, required=True, help="angle of attack (deg)")
    write.add_argument("--beta", type=float, default=0.0, help="sideslip (deg)")
    write.add_argument("--mesh-dir", type=Path, help="the case's mesh (default hpc/runs/<case name>/mesh)")
    write.add_argument("--n-procs", type=int, default=1)
    args = parser.parse_args(argv)

    case = CaseSpec.from_toml(args.case_file)
    mesh_settings = MeshSettings.from_toml(case.mesh, case.mesh_overrides)
    if args.command == "conditions":
        for condition in case.conditions():
            check_condition(condition, mesh_settings)
            print(condition.alpha_deg, condition.beta_deg, condition.name)
        return

    if case.solver is None:
        raise SystemExit(f"{args.case_file} has no solver settings (`solver = ...`)")
    condition = FlowCondition(alpha_deg=args.alpha, beta_deg=args.beta)
    check_condition(condition, mesh_settings)
    mesh_dir = args.mesh_dir or Path("hpc/runs") / args.case_file.stem / "mesh"
    settings = SolverSettings.from_toml(case.solver)
    write_solve_case(case.load_geometry(), settings, condition, mesh_dir, args.solve_dir, args.n_procs)
    velocity = [settings.speed * c for c in freestream_direction(condition)]
    print(f"wrote solve case to {args.solve_dir}")
    print(f"  alpha {condition.alpha_deg:g} deg, beta {condition.beta_deg:g} deg, {settings.speed:g} m/s; "
          f"freestream velocity {foam_vector(velocity)} m/s")


if __name__ == "__main__":
    main()
