"""Tap selection on the extracted CFD dataset (`src/data/cfd.py`).

Turns runs.parquet and the per-run surface tables into the inputs of `select_taps`:
- candidates: faces of an undeflected run's mesh where a tap can go: on the fixed wing (`TAP_PATCH`), on the upper or
  lower skin, thinned to one per `CANDIDATE_SPACING_M` square in planform;
- cp (runs, candidates): every run's Cp interpolated from its own batch's mesh onto the candidates. Candidates that
  are not on every mesh's skin (e.g. opened up by a deflected surface) are dropped;
- targets: the coefficients the controller has to hold at a target (TARGETS: lift, pitch and roll);
- known: the control-surface deflections, which the controller sets itself (only those that vary across runs).

The layout is built in three groups, in this order:
- LEV: N_LEV_STATIONS chordwise rows along the leading-edge vortex footprints, each an inboard/outboard pair per side
  bracketing where the vortex sits, so its spanwise movement shows. Steady CFD fixes the vortices by alpha and beta,
  so a greedy choice never asks for them, but in a gust they lag and break down on their own;
- greedy: N_GREEDY_TAPS chosen for the targets, given the LEV taps;
- uniform: the rest of N_TAPS spread evenly over both skins, for reconstructing the whole Cp field.
Every group is placed in mirror pairs (y -> -y), so the layout is symmetric like the wing. Only the candidate points
are mirrored; the runs are used as solved.

Run as a script to build the layout on every run, estimate how well it explains each target on unseen runs by k-fold
cross-validation, and save it all in a new folder named after the targets:

    python -m src.ml.taps.cfd_taps                          TARGETS -> selections/<YYYYmmdd-HHMMSS>_C_L+C_m+C_l/
    python -m src.ml.taps.cfd_taps --targets C_m            pitch only -> selections/<YYYYmmdd-HHMMSS>_C_m/
    python -m src.ml.taps.cfd_taps --targets C_L C_m --show  also open the figure

Each folder holds taps.parquet and taps.csv (the layout), summary.json (targets, group sizes and the cross-validated
errors) and tap_selection.png. selections/taps.parquet is the layout in use and is never replaced by a run.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.axes import Axes
from torch import Tensor

from src.config import CONFIG
from src.data.cfd import CFD_DATA_DIR, RUNS_FILE, surface_file
from src.ml.atomic import atomic_write_text
from src.ml.taps.tap_selection import (
    N_TAPS,
    CandidatePoints,
    TapSelection,
    farthest_point_fill,
    heldout_curve,
    interpolation,
    select_taps,
)

TAP_PATCH: str = "main_body"  # the only patch that never moves
TARGETS: tuple[str, ...] = ("C_L", "C_m", "C_l")  # lift, pitch and roll: what the controller is to hold at a target;
# the greedy taps weight each equally, after scaling it by its variance across the runs
DEFLECTIONS: str = r"^deflection_.*_deg$"  # runs.parquet columns of the known inputs
UPPER: int = 0
LOWER: int = 1
MIN_SKIN_NZ: float = 0.5  # |nz| below this is an edge or the side wall of a cut-out: no tap there
CANDIDATE_SPACING_M: float = 0.005  # finer than a tap can usefully be placed; sets the problem size
MAX_INTERP_DISTANCE_M: float = 0.003  # a candidate further than this from a mesh's skin is not on it
TAP_MIN_SPACING_M: float = 0.01  # placeholder: closest two taps can be fitted
FOLDS: int = 5  # cross-validation folds: the error estimate uses every run, the final selection uses all of them
SEED: int = 0
USE_UNCONVERGED: bool = True  # also use runs whose loads never settled: their Cp is the last iteration's snapshot,
# but their loads are averaged, so Cp and target need not match for them

# Layout: LEV taps first, then greedy taps for the targets given those, then an even fill up to N_TAPS.
N_LEV_STATIONS: int = 5  # chordwise rows along each leading-edge vortex's footprint; each row is an inboard/outboard
# pair per side, mirrored port/starboard: 4 taps per row, 20 in all

N_GREEDY_TAPS: int = 16  # about where the cross-validated error of a purely greedy layout stops falling; chosen
# in mirror pairs (see with_mirrors), so even
LEV_ALPHA_RANGE_DEG: tuple[float, float] = (5.0, 15.0)  # the vortex moves over this range and breaks down near 20 deg;
# the clean, zero-sideslip runs in it show where its footprint sweeps
LEV_X_RANGE_M: tuple[float, float] = (0.05, 0.30)  # chordwise span of the LEV stations: the wall Cp images show the
# suction strongest from the apex to about mid-chord
LEV_ETA_RANGE: tuple[float, float] = (0.5, 0.98)  # where to look, as a fraction of the local semi-span: on this mesh
# the suction hugs the leading edge
LEV_TAPS_M: tuple[tuple[float, float], ...] | None = None  # (x, y) of starboard LEV taps placed by hand, y > 0,
# mirrored to port; None brackets where the LEV_ALPHA_RANGE_DEG runs' suction peak moves (see lev_stations)
GROUP_COLOURS: dict[str, str] = {"LEV": "orange", "greedy": "red", "actuator": "magenta", "uniform": "white"}

# One folder per run of this script, named after its targets (see `selection_dir`). OUTPUT_DIR/taps.parquet is the
# layout in use (the training scripts' default): a run never replaces it; copy a run's taps.parquet there to adopt it.
OUTPUT_DIR: Path = Path(__file__).parent / "selections"


# ---- Dataset -> selection problem -----------------------------------------------------------------------------------

def load_runs(data_dir: Path = CFD_DATA_DIR, use_unconverged: bool = USE_UNCONVERGED) -> pl.DataFrame:
    """Every run (or every converged run, if not `use_unconverged`), sorted by name."""
    runs = pl.read_parquet(data_dir / RUNS_FILE).sort("run")
    unconverged = runs.filter(~pl.col("converged"))["run"].to_list()
    if use_unconverged:
        if unconverged:
            print(f"using {len(unconverged)} unconverged runs: {unconverged}")
        return runs
    if unconverged:
        print(f"dropping {len(unconverged)} unconverged runs: {unconverged}")
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


@dataclass(frozen=True)
class NoTapZone:
    """A planform rectangle where taps cannot be fitted, on both skins and mirrored port/starboard."""
    name: str
    abs_y: tuple[float, float]
    x: tuple[float, float]


def no_tap_zones() -> list[NoTapZone]:
    """The fuselage, the apex beside it, and the main body further than the allowed distance behind the split flaps
    (from CONFIG.rig)."""
    rig = CONFIG.rig
    inf = float("inf")
    return [
        NoTapZone("fuselage", (0.0, rig.fuselage_half_width_m), (-inf, inf)),
        NoTapZone("apex", (rig.fuselage_half_width_m, inf), (-inf, rig.apex_no_tap_x_m)),
        NoTapZone("behind splitters", rig.splitter_span_m,
                  (rig.splitter_trailing_edge_x_m + rig.max_tap_behind_splitter_m, inf)),
    ]


def tap_allowed(zones: list[NoTapZone]) -> pl.Expr:
    """True where a face lies outside every zone."""
    inside = [pl.col("y").abs().is_between(*z.abs_y, closed="left") & pl.col("x").is_between(*z.x, closed="none")
              for z in zones]
    return ~pl.any_horizontal(inside)


def candidate_points(reference: pl.DataFrame, spacing: float = CANDIDATE_SPACING_M) -> pl.DataFrame:
    """One face per `spacing` square in planform (x, y) on each skin: the one nearest the square's centre. None in
    a `no_tap_zones` zone."""
    return _one_per_square(reference.filter(tap_allowed(no_tap_zones())), spacing)


def no_tap_points(reference: pl.DataFrame, spacing: float = CANDIDATE_SPACING_M) -> pl.DataFrame:
    """`candidate_points`' squares that lie in a `no_tap_zones` zone instead (x, y, surface): where the plots show
    the skin with no taps allowed."""
    return _one_per_square(reference.filter(~tap_allowed(no_tap_zones())), spacing).select("x", "y", "surface")


def _one_per_square(faces: pl.DataFrame, spacing: float) -> pl.DataFrame:
    cells = faces.with_columns((pl.col(c) / spacing).floor().alias(f"cell_{c}") for c in ("x", "y"))
    off_centre = sum(((pl.col(c) / spacing) - pl.col(f"cell_{c}") - 0.5) ** 2 for c in ("x", "y"))
    return (
        cells.sort(off_centre)
        .unique(["surface", "cell_x", "cell_y"], keep="first")
        .sort("surface", "x", "y")
        .select("x", "y", "z", "nx", "ny", "nz", "surface")
    )


def mirror_partners(candidates: pl.DataFrame, spacing: float = CANDIDATE_SPACING_M) -> pl.Series:
    """Index of each candidate's mirror image across the symmetry plane (y -> -y), null if it has none.

    `candidate_points` keeps one candidate per `spacing` square, and the squares are symmetric about y = 0 (square
    j spans [j s, (j + 1) s), so its mirror is square -j - 1). So the mirror is the candidate of the mirrored square
    on the same skin. Candidates within TAP_MIN_SPACING_M / 2 of the centreline are their own mirror: a pair there
    would be two taps closer together than they can be fitted."""
    cells = candidates.with_row_index("index").with_columns(
        cell_x=(pl.col("x") / spacing).floor().cast(pl.Int64), cell_y=(pl.col("y") / spacing).floor().cast(pl.Int64)
    )
    partners = cells.select("surface", "cell_x", cell_y=-pl.col("cell_y") - 1, mirror="index")
    return (
        cells.join(partners, on=["surface", "cell_x", "cell_y"], how="left", maintain_order="left")
        .select(
            pl.when(pl.col("y").abs() < TAP_MIN_SPACING_M / 2).then(pl.col("index")).otherwise(pl.col("mirror"))
        )
        .to_series()
        .alias("mirror")
    )


def with_mirrors(candidates: pl.DataFrame, cp: Tensor) -> tuple[pl.DataFrame, Tensor]:
    """The candidates that have a mirror image, with a `mirror` column (index of their mirror), and their Cp.

    The wing is symmetric, but the candidates also had to be on every run's skin, so a candidate whose mirror was
    dropped goes too. Only the geometry is mirrored: the runs are used as solved."""
    keep = mirror_partners(candidates).is_not_null()
    if n_dropped := int((~keep).sum()):
        print(f"dropping {n_dropped} candidates whose mirror image is not a candidate")
    kept = candidates.filter(keep)
    return kept.with_columns(mirror_partners(kept)), cp[:, torch.from_numpy(keep.to_numpy())]


def as_points(candidates: pl.DataFrame) -> CandidatePoints:
    return CandidatePoints(
        coords=candidates.select("x", "y", "z").to_torch(dtype=pl.Float64),
        surface=candidates["surface"].to_torch(),
        mirror=candidates["mirror"].to_torch() if "mirror" in candidates.columns else None,
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
    no_tap: pl.DataFrame  # x, y, surface of the skin in a no-tap zone, for the plots

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
    reference = skin(reference_run(runs), data_dir)
    candidates = candidate_points(reference)
    cp, covered = candidate_cp(runs, candidates, data_dir)
    if n_dropped := int((~covered).sum()):
        print(f"dropping {n_dropped} of {len(covered)} candidates that are not on every run's skin")
    candidates, cp = with_mirrors(candidates.filter(pl.Series(covered.numpy())), cp[:, covered])
    deflections = runs.select(pl.col(DEFLECTIONS))
    known_names = tuple(c for c in deflections.columns if deflections[c].n_unique() > 1)
    return TapProblem(
        runs=runs,
        candidates=candidates,
        cp=cp,
        targets=runs.select(targets).to_torch(dtype=pl.Float64),
        known=runs.select(known_names).to_torch(dtype=pl.Float64) if known_names else None,
        target_names=targets,
        known_names=known_names,
        no_tap=no_tap_points(reference),
    )


def folds(problem: TapProblem, n_folds: int = FOLDS, seed: int = SEED) -> list[tuple[TapProblem, TapProblem]]:
    """(training, held-out) pairs for k-fold cross-validation: the runs are shuffled and dealt into `n_folds` groups,
    and each group is held out once."""
    order = torch.randperm(problem.runs.height, generator=torch.Generator().manual_seed(seed))
    pairs = []
    for i in range(n_folds):
        held_out = torch.zeros(problem.runs.height, dtype=torch.bool)
        held_out[order[i::n_folds]] = True
        train_rows, test_rows = (~held_out).nonzero().squeeze(1), held_out.nonzero().squeeze(1)
        pairs.append((problem.subset(train_rows), problem.subset(test_rows)))
    return pairs


def cross_validated_curve(problem: TapProblem, lev: Tensor, n_folds: int = FOLDS,
                          choose: LayoutChooser | None = None) -> Tensor:
    """Held-out RMS error / training std against number of taps, in layout order (n_taps + 1, K), over every run.

    Each fold chooses its own taps after the LEV ones (which stay as given) on the other runs, with `choose` (default
    `choose_layout`), and is scored on its held-out runs; the folds' squared errors are pooled, weighted by their
    number of runs. This measures the layout procedure on unseen runs, so a layout made on all runs should do
    slightly better."""
    chooser: LayoutChooser = choose_layout if choose is None else choose
    curves, sizes = [], []
    for train, test in folds(problem, n_folds):
        selection = chooser(train, lev).selection
        curves.append(heldout_curve(selection, train.cp, train.targets, test.cp, test.targets, train.known, test.known))
        sizes.append(test.runs.height)
    length = min(len(c) for c in curves)  # a fold can stop early if its constraints run out of candidates
    squared = torch.stack([c[:length].pow(2) for c in curves])  # (folds, taps + 1, K)
    weights = torch.tensor(sizes, dtype=squared.dtype)[:, None, None] / sum(sizes)
    return (squared * weights).sum(dim=0).sqrt()


# ---- Layout ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TapLayout:
    selection: TapSelection  # every tap, in layout order, with the estimator fitted on all of them
    groups: tuple[str, ...]  # which GROUP_COLOURS group each tap is in, in the same order

    @property
    def group_names(self) -> tuple[str, ...]:
        """The groups this layout has, in layout order."""
        return tuple(dict.fromkeys(self.groups))


type LayoutChooser = Callable[[TapProblem, Tensor], TapLayout]  # (problem, LEV taps) -> the whole layout


def lev_runs(runs: pl.DataFrame) -> list[int]:
    """Rows of the runs with every control surface at 0 deg, no sideslip and alpha in LEV_ALPHA_RANGE_DEG."""
    rows = runs.with_row_index("row").filter(
        pl.all_horizontal(pl.col(DEFLECTIONS) == 0),
        pl.col("beta_deg").abs() < 0.5,
        pl.col("alpha_deg").is_between(*LEV_ALPHA_RANGE_DEG),
    )["row"].to_list()
    if len(rows) < 2:
        raise ValueError(f"need at least two clean, zero-sideslip runs with alpha in {LEV_ALPHA_RANGE_DEG} deg to see "
                         f"the vortices move; found {len(rows)}")
    return rows


def lev_stations(problem: TapProblem) -> list[tuple[float, float]]:
    """(x, y > 0) of the starboard LEV taps: at each of N_LEV_STATIONS stations along LEV_X_RANGE_M, a spanwise pair
    bracketing where the vortex's suction peak sits over the `lev_runs`.

    At each station and in each run, the suction peak is the lowest-Cp upper-skin point within LEV_ETA_RANGE of the
    local semi-span. As alpha rises the peak moves (inboard, on a delta wing); the inboard tap goes at its most
    inboard position and the outboard tap at its most outboard, at least TAP_MIN_SPACING_M apart. Between the two,
    which tap reads the stronger suction says where the vortex is. Averaged over the two sides so the layout is
    symmetric."""
    rows = lev_runs(problem.runs)
    cp = problem.cp[rows]
    upper = problem.candidates.with_row_index("index").filter(pl.col("surface") == UPPER)
    alphas = sorted(problem.runs["alpha_deg"].gather(rows).to_list())
    print(f"LEV taps from {len(rows)} clean runs at alpha {', '.join(f'{a:.1f}' for a in alphas)} deg")

    def peak_track(x: float) -> tuple[float, float] | None:
        """(most inboard, most outboard) |y| of the suction peak over the runs at station x, averaged over the two
        sides, or None if a side has no candidates there."""
        extremes = []
        for sign in (1, -1):
            side = upper.filter((pl.col("x") - x).abs() < CANDIDATE_SPACING_M, pl.col("y") * sign > 0)
            band = side.filter((pl.col("y").abs() / pl.col("y").abs().max()).is_between(*LEV_ETA_RANGE))
            if band.is_empty():
                return None
            peaks = band["y"].abs().to_numpy()[cp[:, band["index"].to_list()].argmin(dim=1).numpy()]
            extremes.append((float(peaks.min()), float(peaks.max())))
        return sum(e[0] for e in extremes) / 2, sum(e[1] for e in extremes) / 2

    # Near the apex the wing can be entirely inside a no-tap zone (the fuselage): start at the first station with
    # candidates on both sides.
    x_start, x_end = LEV_X_RANGE_M
    while peak_track(x_start) is None:
        x_start += CANDIDATE_SPACING_M
        if x_start >= x_end:
            raise ValueError(f"no upper-skin candidates within LEV_ETA_RANGE anywhere in LEV_X_RANGE_M {LEV_X_RANGE_M}")
    if x_start > LEV_X_RANGE_M[0]:
        print(f"LEV stations start at x = {x_start:.3f} m: no candidates further forward (no-tap zones)")
    taps = []
    for x in torch.linspace(x_start, x_end, N_LEV_STATIONS).tolist():
        track = peak_track(x)
        if track is None:
            raise ValueError(f"no upper-skin candidates at x = {x:.3f} m; check LEV_X_RANGE_M")
        inboard, outboard = track
        centre, half = (inboard + outboard) / 2, max((outboard - inboard) / 2, TAP_MIN_SPACING_M / 2)
        print(f"  x = {x:.3f} m: suction peak sweeps |y| {inboard:.3f}-{outboard:.3f} m -> taps at "
              f"y = +-{centre - half:.3f} and +-{centre + half:.3f} m")
        taps += [(x, centre - half), (x, centre + half)]
    return taps


def lev_taps(problem: TapProblem) -> Tensor:
    """Candidate indices of the LEV taps, starboard then port for each: LEV_TAPS_M if given, else bracketing the
    vortex's suction peak (`lev_stations`). Each starboard tap is snapped to the nearest unused upper-skin candidate in
    planform and the port tap is its mirror image."""
    points = list(LEV_TAPS_M) if LEV_TAPS_M is not None else lev_stations(problem)
    upper = problem.candidates.with_row_index("index").filter(pl.col("surface") == UPPER, pl.col("y") > 0)
    taps: list[int] = []
    for x, y in points:
        free = upper.filter(~pl.col("index").is_in(taps))  # a close pair must not snap onto one candidate
        starboard = free.sort((pl.col("x") - x) ** 2 + (pl.col("y") - y) ** 2).row(0, named=True)
        taps += [starboard["index"], starboard["mirror"]]
    return torch.tensor(taps, dtype=torch.long)


def choose_layout(problem: TapProblem, lev: Tensor, n_greedy: int = N_GREEDY_TAPS, n_taps: int = N_TAPS) -> TapLayout:
    """The `lev` taps, then `n_greedy` taps chosen for the targets given them, then an even fill up to `n_taps`."""
    if len(lev) + n_greedy > n_taps:
        raise ValueError(f"{len(lev)} LEV + {n_greedy} greedy taps is more than the {n_taps} available")
    greedy = select_taps(
        problem.cp, problem.targets, problem.points, n_taps=len(lev) + n_greedy, known=problem.known,
        min_spacing=TAP_MIN_SPACING_M, fixed=lev,
    )
    fill = farthest_point_fill(problem.points, greedy.taps, n_taps - len(greedy.taps))
    taps = torch.cat([greedy.taps, fill])
    selection = select_taps(  # scores the whole layout in this order: its unexplained curve and estimator
        problem.cp, problem.targets, problem.points, n_taps=len(taps), known=problem.known, fixed=taps,
    )
    groups = ("LEV",) * len(lev) + ("greedy",) * (len(greedy.taps) - len(lev)) + ("uniform",) * len(fill)
    return TapLayout(selection=selection, groups=groups)


def tap_table(problem: TapProblem, layout: TapLayout) -> pl.DataFrame:
    """The taps in layout order, with their group, position and the share of each target unexplained once added."""
    selection = layout.selection
    taps = problem.candidates.select(pl.all().gather(selection.taps.tolist()))
    unexplained = {
        f"unexplained_{name}": selection.unexplained[1:, k].numpy() for k, name in enumerate(problem.target_names)
    }
    return taps.with_row_index("order").with_columns(
        group=pl.Series(layout.groups), **{k: pl.Series(v) for k, v in unexplained.items()}
    )


# ---- Report ---------------------------------------------------------------------------------------------------------

def _planform(ax: Axes, problem: TapProblem, taps: pl.DataFrame, surface: int, title: str,
              groups: tuple[str, ...]) -> None:
    """Candidates coloured by how much their Cp varies across runs, with the taps numbered in layout order and
    outlined in their group's colour."""
    on_surface = (problem.candidates["surface"] == surface).to_numpy()
    spread = problem.cp[:, on_surface].std(dim=0).numpy() if problem.runs.height > 1 else None
    xy = problem.candidates.filter(pl.Series(on_surface))
    scatter = ax.scatter(xy["y"], xy["x"], c=spread, s=2, cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="Cp std across runs")
    blocked = problem.no_tap.filter(pl.col("surface") == surface)  # the skin in a no-tap zone, in its own shape
    ax.scatter(blocked["y"], blocked["x"], s=2, color="0.55", label="no taps")
    for group in groups:
        colour = GROUP_COLOURS[group]
        picked = taps.filter(pl.col("surface") == surface, pl.col("group") == group)
        ax.scatter(picked["y"], picked["x"], s=30, facecolors="none", edgecolors=colour, linewidths=1.2,
                   label=f"{group} ({taps.filter(pl.col('group') == group).height})")
        for order, y, x in picked.select("order", "y", "x").iter_rows():
            ax.annotate(str(order), (y, x), fontsize=6, color=colour, xytext=(3, 3), textcoords="offset points")
    ax.set_facecolor("0.85")  # so the white uniform taps show off the wing
    ax.set_aspect("equal")
    ax.invert_yaxis()  # nose at the top
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)


def plot_layout(problem: TapProblem, layout: TapLayout, path: Path, heldout: Tensor | None = None,
                show: bool = False) -> None:
    """Tap layout on both skins, and the unexplained target variance against number of taps in layout order: on the
    runs the layout was made on and, if `heldout` (held-out RMS error / training std per tap count) is given, on
    unseen runs. Dotted lines mark where each group of taps ends. Saved to `path`."""
    selection = layout.selection
    taps = tap_table(problem, layout)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    _planform(axes[0], problem, taps, UPPER, "Upper skin (seen from above)", layout.group_names)
    _planform(axes[1], problem, taps, LOWER, "Lower skin (seen from above)", layout.group_names)
    for k, name in enumerate(problem.target_names):
        line, = axes[2].semilogy(selection.unexplained[:, k].numpy(), label=f"{name} training")
        if heldout is not None:
            # (RMS error / training std)^2 is the held-out counterpart of the training variance fraction
            axes[2].semilogy(heldout[:, k].pow(2).numpy(), "--", color=line.get_color(), label=f"{name} cross-validated")
    end = 0
    for group in layout.group_names:
        end += layout.groups.count(group)
        axes[2].axvline(end, color="0.5", linestyle=":", linewidth=1)
        axes[2].text(end, 1.0, f" {group} ", transform=axes[2].get_xaxis_transform(), ha="right", va="top", fontsize=8)
    axes[2].set_xlabel("Taps (layout order)")
    axes[2].set_ylabel("Unexplained fraction of target variance")
    axes[2].legend(loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    handles, labels = axes[0].get_legend_handles_labels()  # every group, even one with no tap on the upper skin
    fig.legend(handles, labels, loc="upper center", ncols=len(handles), title="Tap groups (count)",
               facecolor="0.85")  # grey, as the planform backgrounds, so the white uniform marker shows
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def selection_dir(targets: tuple[str, ...], tag: str = "", root: Path = OUTPUT_DIR) -> Path:
    """root/<YYYYmmdd-HHMMSS>_<targets joined by +>[_<tag>], e.g. selections/20261125-150212_C_L+C_m+C_l, created."""
    folder = root / f"{datetime.now():%Y%m%d-%H%M%S}_{'+'.join(targets)}{f'_{tag}' if tag else ''}"
    folder.mkdir(parents=True)
    return folder


@dataclass(frozen=True)
class SelectionRun:
    folder: Path  # where it was saved
    problem: TapProblem
    layout: TapLayout
    taps: pl.DataFrame  # tap_table of the layout


def run_selection(targets: tuple[str, ...] = TARGETS, choose: LayoutChooser = choose_layout, tag: str = "",
                  show: bool = False) -> SelectionRun:
    """Build the layout for `targets` on every run with `choose`, cross-validate the procedure, and save it in a new
    folder of OUTPUT_DIR (`selection_dir`, with `tag` in its name): taps.parquet and taps.csv (the layout),
    summary.json (the settings and the cross-validated errors after each group) and tap_selection.png. Never replaces
    OUTPUT_DIR/taps.parquet, the layout in use."""
    problem = tap_problem(targets=targets)
    print(
        f"{problem.runs.height} runs, {problem.candidates.height} candidates, "
        f"targets {problem.target_names}, known {problem.known_names}"
    )
    lev = lev_taps(problem)
    layout = choose(problem, lev)  # the layout to build: every run, for full coverage
    taps = tap_table(problem, layout)
    with pl.Config(tbl_rows=-1):
        print(taps)
    heldout = cross_validated_curve(problem, lev, choose=choose)
    errors: dict[str, dict[str, float]] = {}
    for k, name in enumerate(problem.target_names):
        end = 0
        errors[name] = {}
        for group in layout.group_names:
            end += layout.groups.count(group)
            errors[name][f"after_{group}_{end}_taps"] = float(heldout[end, k])
            print(f"{name} {FOLDS}-fold cross-validated RMS error / training std after the {group} taps "
                  f"({end} taps): {heldout[end, k]:.3f}")

    folder = selection_dir(problem.target_names, tag)
    taps.write_parquet(folder / "taps.parquet")
    taps.write_csv(folder / "taps.csv")
    atomic_write_text(folder / "summary.json", json.dumps({
        "targets": list(problem.target_names),
        "known": list(problem.known_names),
        "n_runs": problem.runs.height,
        "n_candidates": problem.candidates.height,
        "groups": {group: layout.groups.count(group) for group in layout.group_names},
        "folds": FOLDS,
        "cross_validated_rms_error_over_std": errors,
    }, indent=2))
    plot_layout(problem, layout, folder / "tap_selection.png", heldout, show)
    print(f"saved {folder}")
    return SelectionRun(folder, problem, layout, taps)


def main(targets: tuple[str, ...] = TARGETS, show: bool = False) -> Path:
    """LEV, greedy and uniform taps (`choose_layout`) for `targets`, saved by `run_selection`. Returns the folder."""
    return run_selection(targets, choose_layout, show=show).folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a tap layout on the CFD runs and save it in a new folder of "
                                                 f"{OUTPUT_DIR}, named after the targets.")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS),
                        help=f"runs.parquet coefficients the layout is chosen for (default: {' '.join(TARGETS)})")
    parser.add_argument("--show", action="store_true", help="also open the figure")
    args = parser.parse_args()
    main(tuple(args.targets), args.show)
