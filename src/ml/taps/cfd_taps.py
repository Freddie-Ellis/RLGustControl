"""Tap selection on the extracted CFD dataset (`src/data/cfd.py`).

Turns runs.parquet and the per-run surface tables into the inputs of `select_taps`:
- candidates: faces of an undeflected run's mesh where a tap can go: on the fixed wing (`TAP_PATCH`), on the upper or
  lower skin, thinned to one per `CANDIDATE_SPACING_M` square in planform;
- cp (runs, candidates): every run's Cp interpolated from its own batch's mesh onto the candidates. Candidates that
  are not on every mesh's skin (e.g. opened up by a deflected surface) are dropped;
- targets: the moment coefficients the controller has to hold;
- known: the control-surface deflections, which the controller sets itself (only those that vary across runs).

Run as a script to select taps on a training split, report how much of each target they explain on held-out runs,
and save the layout and a plot.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.axes import Axes
from torch import Tensor

from src.data.cfd import CFD_DATA_DIR, RUNS_FILE, surface_file
from src.ml.taps.tap_selection import (
    N_TAPS,
    CandidatePoints,
    TapSelection,
    heldout_curve,
    interpolation,
    select_taps,
)

TAP_PATCH: str = "main_body"  # the only patch that never moves
TARGETS: tuple[str, ...] = ("C_m",)  # pitch moment coefficients
DEFLECTIONS: str = r"^deflection_.*_deg$"  # runs.parquet columns of the known inputs
UPPER: int = 0
LOWER: int = 1
MIN_SKIN_NZ: float = 0.5  # |nz| below this is an edge or the side wall of a cut-out: no tap there
CANDIDATE_SPACING_M: float = 0.005  # finer than a tap can usefully be placed; sets the problem size
MAX_INTERP_DISTANCE_M: float = 0.003  # a candidate further than this from a mesh's skin is not on it
TAP_MIN_SPACING_M: float = 0.01  # placeholder: closest two taps can be fitted
HOLDOUT_FRACTION: float = 0.2
SEED: int = 0

OUTPUT_DIR: Path = Path(__file__).parent / "selections"
PLOTS_DIR: Path = Path(__file__).parent / "plots"


# ---- Dataset -> selection problem -----------------------------------------------------------------------------------

def load_runs(data_dir: Path = CFD_DATA_DIR) -> pl.DataFrame:
    """Every converged run, sorted by name."""
    runs = pl.read_parquet(data_dir / RUNS_FILE).sort("run")
    if dropped := runs.filter(~pl.col("converged"))["run"].to_list():
        print(f"dropping {len(dropped)} unconverged runs: {dropped}")
    return runs.filter(pl.col("converged"))


def skin(run: str, data_dir: Path = CFD_DATA_DIR) -> pl.DataFrame:
    """Faces of one run where a tap could go, with a `surface` id: `UPPER` or `LOWER`."""
    return (
        pl.scan_parquet(surface_file(run, data_dir))
        .filter(pl.col("patch") == TAP_PATCH, pl.col("nz").abs() >= MIN_SKIN_NZ)
        .with_columns(surface=pl.when(pl.col("nz") > 0).then(UPPER).otherwise(LOWER).cast(pl.Int64))
        .collect()
    )


def reference_run(runs: pl.DataFrame) -> str:
    """A run with every control surface at 0 deg: its mesh supplies the candidate points."""
    clean = runs.filter(pl.all_horizontal(pl.col(DEFLECTIONS) == 0))
    if clean.is_empty():
        raise ValueError("no run has every control surface at 0 deg to take the candidate points from")
    return clean["run"][0]


def candidate_points(reference: pl.DataFrame, spacing: float = CANDIDATE_SPACING_M) -> pl.DataFrame:
    """One face per `spacing` square in planform (x, y) on each skin: the one nearest the square's centre."""
    cells = reference.with_columns((pl.col(c) / spacing).floor().alias(f"cell_{c}") for c in ("x", "y"))
    off_centre = sum(((pl.col(c) / spacing) - pl.col(f"cell_{c}") - 0.5) ** 2 for c in ("x", "y"))
    return (
        cells.sort(off_centre)
        .unique(["surface", "cell_x", "cell_y"], keep="first")
        .sort("surface", "x", "y")
        .select("x", "y", "z", "nx", "ny", "nz", "surface")
    )


def as_points(candidates: pl.DataFrame) -> CandidatePoints:
    return CandidatePoints(
        coords=candidates.select("x", "y", "z").to_torch(dtype=pl.Float64),
        surface=candidates["surface"].to_torch(),
    )


def candidate_cp(runs: pl.DataFrame, candidates: pl.DataFrame, data_dir: Path = CFD_DATA_DIR) -> tuple[Tensor, Tensor]:
    """Cp at the candidates for every run (N, C), and whether each candidate is on every run's skin (C,).

    The runs of a batch share one mesh, so the interpolation is built once per batch.
    """
    points = as_points(candidates)
    cp = torch.empty(runs.height, candidates.height, dtype=torch.float64)
    covered = torch.ones(candidates.height, dtype=torch.bool)
    for _, batch in runs.with_row_index("row").group_by("batch", maintain_order=True):
        names = batch["run"].to_list()
        mesh = skin(names[0], data_dir)
        interp = interpolation(mesh.select("x", "y", "z").to_torch(dtype=pl.Float64), mesh["surface"].to_torch(), points)
        covered &= interp.nearest <= MAX_INTERP_DISTANCE_M
        for row, run in zip(batch["row"].to_list(), names, strict=True):
            faces = skin(run, data_dir)
            if faces.height != mesh.height:
                raise ValueError(f"{run}: {faces.height} skin faces, but its batch's mesh has {mesh.height}")
            cp[row] = interp(faces["cp"].to_torch())
    return cp, covered


@dataclass(frozen=True)
class TapProblem:
    """Everything `select_taps` needs, with the names and positions to report the result in."""

    runs: pl.DataFrame  # N rows of runs.parquet
    candidates: pl.DataFrame  # C rows: x, y, z, nx, ny, nz, surface
    cp: Tensor  # (N, C)
    targets: Tensor  # (N, K)
    known: Tensor | None  # (N, J), None if no known input varies
    target_names: tuple[str, ...]
    known_names: tuple[str, ...]

    @property
    def points(self) -> CandidatePoints:
        return as_points(self.candidates)

    def subset(self, rows: Tensor) -> "TapProblem":
        """The same problem on only these runs."""
        return replace(
            self,
            runs=self.runs[rows.numpy()],
            cp=self.cp[rows],
            targets=self.targets[rows],
            known=None if self.known is None else self.known[rows],
        )


def tap_problem(data_dir: Path = CFD_DATA_DIR, targets: tuple[str, ...] = TARGETS) -> TapProblem:
    runs = load_runs(data_dir)
    candidates = candidate_points(skin(reference_run(runs), data_dir))
    cp, covered = candidate_cp(runs, candidates, data_dir)
    if n_dropped := int((~covered).sum()):
        print(f"dropping {n_dropped} of {len(covered)} candidates that are not on every run's skin")
    deflections = runs.select(pl.col(DEFLECTIONS))
    known_names = tuple(c for c in deflections.columns if deflections[c].n_unique() > 1)
    return TapProblem(
        runs=runs,
        candidates=candidates.filter(pl.Series(covered.numpy())),
        cp=cp[:, covered],
        targets=runs.select(targets).to_torch(dtype=pl.Float64),
        known=runs.select(known_names).to_torch(dtype=pl.Float64) if known_names else None,
        target_names=targets,
        known_names=known_names,
    )


def split(problem: TapProblem, holdout: float = HOLDOUT_FRACTION, seed: int = SEED) -> tuple[TapProblem, TapProblem]:
    """(training, held-out) runs, chosen at random."""
    order = torch.randperm(problem.runs.height, generator=torch.Generator().manual_seed(seed))
    n_test = round(holdout * problem.runs.height)
    return problem.subset(order[n_test:].sort().values), problem.subset(order[:n_test].sort().values)


def choose_taps(problem: TapProblem, n_taps: int = N_TAPS) -> TapSelection:
    return select_taps(
        problem.cp,
        problem.targets,
        problem.points,
        n_taps=n_taps,
        known=problem.known,
        min_spacing=TAP_MIN_SPACING_M,
    )


def tap_table(problem: TapProblem, selection: TapSelection) -> pl.DataFrame:
    """The taps in the order chosen, with their position and the share of each target unexplained once it is added."""
    taps = problem.candidates.select(pl.all().gather(selection.taps.tolist()))
    unexplained = {
        f"unexplained_{name}": selection.unexplained[1:, k].numpy() for k, name in enumerate(problem.target_names)
    }
    return taps.with_row_index("order").with_columns(**{k: pl.Series(v) for k, v in unexplained.items()})


# ---- Report ---------------------------------------------------------------------------------------------------------

def _planform(ax: Axes, problem: TapProblem, taps: pl.DataFrame, surface: int, title: str) -> None:
    """Candidates coloured by how much their Cp varies across runs, with the taps numbered in the order chosen."""
    on_surface = (problem.candidates["surface"] == surface).to_numpy()
    spread = problem.cp[:, on_surface].std(dim=0).numpy() if problem.runs.height > 1 else None
    xy = problem.candidates.filter(pl.Series(on_surface))
    scatter = ax.scatter(xy["y"], xy["x"], c=spread, s=2, cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="Cp std across runs")
    picked = taps.filter(pl.col("surface") == surface)
    ax.scatter(picked["y"], picked["x"], s=30, facecolors="none", edgecolors="red")
    for order, y, x in picked.select("order", "y", "x").iter_rows():
        ax.annotate(str(order), (y, x), fontsize=6, color="red", xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # nose at the top
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)


def plot_selection(problem: TapProblem, selection: TapSelection, heldout: Tensor | None = None) -> None:
    """Tap layout on both skins, and the unexplained target variance against number of taps: on the training runs
    and, if `heldout` (from `heldout_curve`) is given, on the held-out runs."""
    taps = tap_table(problem, selection)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    _planform(axes[0], problem, taps, UPPER, "Upper skin (seen from above)")
    _planform(axes[1], problem, taps, LOWER, "Lower skin (seen from above)")
    for k, name in enumerate(problem.target_names):
        line, = axes[2].semilogy(selection.unexplained[:, k].numpy(), label=f"{name} training")
        if heldout is not None:
            # (RMS error / training std)^2 is the held-out counterpart of the training variance fraction
            axes[2].semilogy(heldout[:, k].pow(2).numpy(), "--", color=line.get_color(), label=f"{name} held out")
    axes[2].set_xlabel("Taps")
    axes[2].set_ylabel("Unexplained fraction of target variance")
    axes[2].legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / "tap_selection.png", dpi=200)
    plt.show()


def main() -> None:
    problem = tap_problem()
    train, test = split(problem)
    print(
        f"{problem.runs.height} runs ({train.runs.height} training, {test.runs.height} held out), "
        f"{problem.candidates.height} candidates, targets {problem.target_names}, known {problem.known_names}"
    )
    selection = choose_taps(train)
    taps = tap_table(train, selection)
    with pl.Config(tbl_rows=-1):
        print(taps)
    heldout = None
    if test.runs.height:
        heldout = heldout_curve(
            selection, train.cp, train.targets, test.cp, test.targets, train.known, test.known
        )
        for k, name in enumerate(problem.target_names):
            best = int(heldout[:, k].argmin())
            print(
                f"{name} held-out RMS error / training std: {heldout[-1, k]:.3f} with all {len(selection.taps)} taps, "
                f"lowest {heldout[best, k]:.3f} with the first {best}"
            )
    OUTPUT_DIR.mkdir(exist_ok=True)
    taps.write_parquet(OUTPUT_DIR / "taps.parquet")
    plot_selection(train, selection, heldout)


if __name__ == "__main__":
    main()
