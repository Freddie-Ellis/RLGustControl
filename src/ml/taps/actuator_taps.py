"""Tap selection with an actuator group in place of the uniform fill: every control surface gets taps that see it.

    python -m src.ml.taps.actuator_taps                            targets from cfd_taps.TARGETS
    python -m src.ml.taps.actuator_taps --targets C_m              pitch only
    python -m src.ml.taps.actuator_taps --n-actuator 8 --show      a smaller actuator group (the rest uniform)

The layout is built as in src/ml/taps/cfd_taps.py up to its greedy taps, then:
- LEV: N_LEV_STATIONS rows along the leading-edge vortex footprints (as cfd_taps);
- greedy: N_GREEDY_TAPS chosen for the targets, given the LEV taps, with the deflections known (as cfd_taps);
- actuator: N_ACTUATOR_TAPS chosen greedily to estimate the deflections of every actuator varied in the runs from the
  taps alone, given the taps so far, with nothing else known: alpha, beta and the other deflections vary freely, as
  in a gust. Each deflection is scaled by its variance and weighted equally, so the group spends its taps where they
  explain the most deflection variance in total; one surface that is already well seen gains little from another
  tap, so they spread over the surfaces. Chosen in mirror pairs like every group (6 pairs for 12 taps);
- uniform: whatever of N_TAPS is left, spread evenly (none with the defaults: 20 + 16 + 12 = 48).

Saved like cfd_taps, in selections/<YYYYmmdd-HHMMSS>_<targets>_actuators/: taps.parquet, taps.csv, summary.json and
tap_selection.png, plus the actuator observability check of the new layout (src/ml/taps/actuator_observability.py):
actuator_observability.csv, actuator_observability.png and actuator_signal.png.
"""

import argparse

import torch
from torch import Tensor

from src.ml.taps import actuator_observability
from src.ml.taps.cfd_taps import (
    N_GREEDY_TAPS,
    TAP_MIN_SPACING_M,
    TARGETS,
    TapLayout,
    TapProblem,
    run_selection,
)
from src.ml.taps.tap_selection import N_TAPS, farthest_point_fill, select_taps

N_ACTUATOR_TAPS: int = 12  # taps chosen to see the actuators, in place of the uniform fill


def choose_actuator_layout(problem: TapProblem, lev: Tensor, n_greedy: int = N_GREEDY_TAPS,
                           n_actuator: int = N_ACTUATOR_TAPS, n_taps: int = N_TAPS) -> TapLayout:
    """The `lev` taps, `n_greedy` taps for the targets given them, `n_actuator` taps for the deflections given all of
    those, then an even fill up to `n_taps` (see the module docstring)."""
    if problem.known is None:
        raise ValueError("no deflection varies across the runs: there are no actuators to choose taps for")
    if len(lev) + n_greedy + n_actuator > n_taps:
        raise ValueError(f"{len(lev)} LEV + {n_greedy} greedy + {n_actuator} actuator taps is more than the {n_taps} "
                         "available")
    greedy = select_taps(
        problem.cp, problem.targets, problem.points, n_taps=len(lev) + n_greedy, known=problem.known,
        min_spacing=TAP_MIN_SPACING_M, fixed=lev,
    )
    actuator = select_taps(  # the deflections are the targets here, so nothing is known
        problem.cp, problem.known, problem.points, n_taps=len(greedy.taps) + n_actuator,
        min_spacing=TAP_MIN_SPACING_M, fixed=greedy.taps,
    )
    fill = farthest_point_fill(problem.points, actuator.taps, n_taps - len(actuator.taps))
    taps = torch.cat([actuator.taps, fill])
    selection = select_taps(  # scores the whole layout for the targets in this order: its unexplained curve
        problem.cp, problem.targets, problem.points, n_taps=len(taps), known=problem.known, fixed=taps,
    )
    groups = (("LEV",) * len(lev) + ("greedy",) * (len(greedy.taps) - len(lev))
              + ("actuator",) * (len(actuator.taps) - len(greedy.taps)) + ("uniform",) * len(fill))
    return TapLayout(selection=selection, groups=groups)


def main(targets: tuple[str, ...] = TARGETS, n_actuator: int = N_ACTUATOR_TAPS, show: bool = False) -> None:
    def choose(problem: TapProblem, lev: Tensor) -> TapLayout:
        return choose_actuator_layout(problem, lev, n_actuator=n_actuator)

    run = run_selection(targets, choose, tag="actuators", show=show)
    print("\nHow well the new layout sees each actuator:")
    actuator_observability.report(run.taps, run.problem.runs, run.problem.cp[:, run.layout.selection.taps],
                                  run.folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a tap layout with an actuator group in place of the uniform "
                                                 "fill, and check how well it sees each actuator.")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS),
                        help=f"runs.parquet coefficients the greedy taps are chosen for (default: {' '.join(TARGETS)})")
    parser.add_argument("--n-actuator", type=int, default=N_ACTUATOR_TAPS,
                        help=f"taps in the actuator group (default {N_ACTUATOR_TAPS}); any left of the {N_TAPS} are "
                             "spread evenly")
    parser.add_argument("--show", action="store_true", help="also open the layout figure")
    args = parser.parse_args()
    main(tuple(args.targets), args.n_actuator, args.show)
