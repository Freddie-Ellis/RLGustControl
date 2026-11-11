"""Write CFD case files for a table of cases, e.g. the design in src/ml/taps/cfd_cases/cfd_cases1.csv.

    uv run python -m src.cfd.batch_cases src/ml/taps/cfd_cases/cfd_cases1.csv configs/cases/batch1

The table has one row per CFD case: `case` (an id), actuator_0_deg ... actuator_11_deg, aoa_deg and
sideslip_deg. Rows with the same actuator angles share a mesh, so they become one case file (one mesh) solved at
each of their (alpha, beta) pairs. Case files are named <folder name>_<first case id>, e.g. batch1_036.toml, and
index.csv in the folder maps every row's case id to its case file and condition folder, i.e. to its results in
hpc/runs/<case file>/solve/<condition>/results.json. Submit them all with
    bash hpc/submit_cases.sh configs/cases/batch1

Every row is checked against the geometry's deflection limits and the mesh's angle range before anything is
written; if any fails, nothing is written and the failures are listed.
"""

import argparse
from pathlib import Path

import polars as pl

from src.cfd.case import FlowCondition
from src.cfd.geometry import GeometryConfig
from src.cfd.mesh import MeshSettings
from src.cfd.solve import check_condition

# actuator_<k>_deg column -> control surface in the geometry (the order of src/config.py's rig surfaces).
ACTUATOR_SURFACES: tuple[str, ...] = (
    "wingtip_r", "wingtip_l", "le_flap_r", "le_flap_l", "obte_flap_r", "obte_flap_l", "ibte_flap_r",
    "ibte_flap_l", "upr_splitter_flap_r", "upr_splitter_flap_l", "lwr_splitter_flap_r", "lwr_splitter_flap_l",
)
ACTUATOR_COLUMNS: list[str] = [f"actuator_{k}_deg" for k in range(len(ACTUATOR_SURFACES))]


def check_rows(table: pl.DataFrame, geometry: GeometryConfig, mesh: MeshSettings) -> list[str]:
    """One message per row that the geometry or the mesh can't take."""
    problems: list[str] = []
    for row in table.iter_rows(named=True):
        deflections = {s: row[c] for s, c in zip(ACTUATOR_SURFACES, ACTUATOR_COLUMNS)}
        try:
            geometry.check_deflections(deflections)
            check_condition(FlowCondition(alpha_deg=row["aoa_deg"], beta_deg=row["sideslip_deg"]), mesh)
        except ValueError as e:
            problems.append(f"case {row['case']}: {e}")
    return problems


def case_file(name: str, ids: list[int], deflections: dict[str, float], pairs: list[tuple[float, float]],
              geometry: Path, mesh: Path, solver: Path) -> str:
    lines = [
        f"# Written by src/cfd/batch_cases.py: rows {', '.join(map(str, ids))} of the case table (see index.csv).",
        "",
        f'description = "Batch case {name}: {len(pairs)} flow condition(s)"',
        f'geometry = "{geometry.as_posix()}"',
        f'mesh = "{mesh.as_posix()}"',
        f'solver = "{solver.as_posix()}"',
        "",
        "[deflections_deg]",
        *(f"{surface} = {angle:g}" for surface, angle in deflections.items() if angle != 0),
        "",
        "[flow]",
        "pairs = [" + ", ".join(f"[{a:g}, {b:g}]" for a, b in pairs) + "]",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write CFD case files for a table of cases.")
    parser.add_argument("table", type=Path, help="CSV with case, actuator_<k>_deg, aoa_deg, sideslip_deg")
    parser.add_argument("out_dir", type=Path, help="folder for the case files, e.g. configs/cases/batch1")
    parser.add_argument("--geometry", type=Path, default=Path("cad/uav/geometry.toml"))
    parser.add_argument("--mesh", type=Path, default=Path("configs/mesh/uav.toml"))
    parser.add_argument("--solver", type=Path, default=Path("configs/solver/uav_rans.toml"))
    args = parser.parse_args(argv)

    table = pl.read_csv(args.table)
    problems = check_rows(table, GeometryConfig.from_toml(args.geometry), MeshSettings.from_toml(args.mesh))
    if problems:
        raise SystemExit(f"{len(problems)} of {table.height} rows can't be run; nothing written:\n  "
                         + "\n  ".join(problems[:20]) + ("\n  ..." if len(problems) > 20 else ""))

    batch = args.out_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    groups = table.sort("case").partition_by(ACTUATOR_COLUMNS, maintain_order=True)
    for rows in groups:
        ids = rows["case"].to_list()
        name = f"{batch}_{ids[0]:03d}"
        pairs = list(zip(rows["aoa_deg"].to_list(), rows["sideslip_deg"].to_list()))
        deflections = {s: rows[c][0] for s, c in zip(ACTUATOR_SURFACES, ACTUATOR_COLUMNS)}
        (args.out_dir / f"{name}.toml").write_text(
            case_file(name, ids, deflections, pairs, args.geometry, args.mesh, args.solver))
        index += [{"case": i, "case_name": name, "condition": FlowCondition(alpha_deg=a, beta_deg=b).name}
                  for i, (a, b) in zip(ids, pairs)]
    pl.DataFrame(index).write_csv(args.out_dir / "index.csv")
    print(f"wrote {len(groups)} case files ({table.height} conditions) and index.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
