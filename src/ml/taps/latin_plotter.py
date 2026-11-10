from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from src.ml.taps.latin_hyper import CASES_DIR, DESIGN_INPUTS, DesignInput

CONTOUR_RESOLUTION: int = 80
PLOTS_DIR: Path = Path(__file__).parent / "plots"
cases_name = "batch_2"

INPUTS: dict[str, DesignInput] = {i.name: i for i in DESIGN_INPUTS}


def _unit(values: np.ndarray, name: str) -> np.ndarray:
    """Values scaled to [0, 1] over the input's design range, so distances compare across inputs."""
    spec = INPUTS[name]
    return (values - spec.lower) / (spec.upper - spec.lower)


def _from_unit(values: np.ndarray, name: str) -> np.ndarray:
    spec = INPUTS[name]
    return spec.lower + values * (spec.upper - spec.lower)


def _nearest_distance_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance (in [0, 1] units) from each point of a 2D mesh to the nearest case in this projection."""
    axis = np.linspace(0.0, 1.0, CONTOUR_RESOLUTION)
    gx, gy = np.meshgrid(axis, axis)
    mesh = np.stack([gx.ravel(), gy.ravel()], axis=1)
    points = np.stack([x, y], axis=1)
    dist = np.linalg.norm(mesh[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    return gx, gy, dist.reshape(gx.shape)


def _coverage_contour(ax: Axes, cases: pl.DataFrame, x_name: str, y_name: str, levels: np.ndarray) -> None:
    x, y = cases[x_name].to_numpy(), cases[y_name].to_numpy()
    gx, gy, dist = _nearest_distance_field(_unit(x, x_name), _unit(y, y_name))
    ax.contourf(_from_unit(gx, x_name), _from_unit(gy, y_name), dist, levels=levels, cmap="magma_r", extend="max")
    ax.scatter(x, y, s=3, c="white", edgecolors="none")


def _short_label(name: str) -> str:
    return name.removesuffix("_deg").replace("actuator_", "A").replace("sideslip", "beta")


def plot_pair_grid(cases: pl.DataFrame) -> None:
    """Pair grid over every design input.

    Lower: cases (orange = structured, blue = LHS). Upper: distance to the nearest case, in units of
    each input's range. Diagonal: 1D coverage of each input.
    """
    names = list(INPUTS)
    is_lhs = (cases["kind"] == "lhs").to_numpy()
    # shared levels so gaps are comparable between panels; capped at a fifth of the range
    levels = np.linspace(0.0, 0.2, 12)
    n_dims = len(names)
    fig = plt.figure(figsize=(18, 18))
    axes: np.ndarray = np.asarray(fig.subplots(n_dims, n_dims, squeeze=False))
    for i, y_name in enumerate(names):
        for j, x_name in enumerate(names):
            ax: Axes = axes[i, j]
            x_spec = INPUTS[x_name]
            if i == j:
                ax.hist(cases[x_name].to_numpy(), bins=10, range=(x_spec.lower, x_spec.upper), color="tab:blue")
            elif i > j:
                x, y = cases[x_name].to_numpy(), cases[y_name].to_numpy()
                ax.scatter(x[is_lhs], y[is_lhs], s=2, c="tab:blue")
                ax.scatter(x[~is_lhs], y[~is_lhs], s=2, c="tab:orange")
            else:
                _coverage_contour(ax, cases, x_name, y_name, levels)
            ax.set_xlim(x_spec.lower, x_spec.upper)
            if i != j:
                ax.set_ylim(INPUTS[y_name].lower, INPUTS[y_name].upper)
            ax.tick_params(labelsize=5)
            if j > 0:
                ax.set_yticklabels([])
            if i < n_dims - 1:
                ax.set_xticklabels([])
            if i == n_dims - 1:
                ax.set_xlabel(_short_label(x_name), fontsize=8)
            if j == 0:
                ax.set_ylabel(_short_label(y_name), fontsize=8)
    fig.suptitle("Pairwise projections (lower: cases, orange = structured; upper: distance to nearest case; diag: 1D coverage)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    plt.savefig(PLOTS_DIR / f"{cases_name}_pair_grid.png", dpi=300)
    plt.show()


def plot_pair_contour(cases: pl.DataFrame, x_name: str, y_name: str) -> None:
    """Nearest-case distance contour for one pair of inputs, showing projected coverage gaps."""
    _, ax = plt.subplots(figsize=(8, 7))
    x, y = cases[x_name].to_numpy(), cases[y_name].to_numpy()
    gx, gy, dist = _nearest_distance_field(_unit(x, x_name), _unit(y, y_name))
    contour = ax.contourf(_from_unit(gx, x_name), _from_unit(gy, y_name), dist, levels=20, cmap="magma_r")
    plt.colorbar(contour, label="Distance to nearest case (fraction of input range)")
    ax.scatter(x, y, s=12, c="white", edgecolors="black", linewidths=0.5)
    ax.set_xlabel(f"{x_name} (deg)")
    ax.set_ylabel(f"{y_name} (deg)")
    ax.set_title(f"Coverage of {x_name} vs {y_name}")
    plt.savefig(PLOTS_DIR / f"{cases_name}_{x_name}_{y_name}_coverage_contour.png", dpi=300)
    plt.show()


def main() -> None:
    cases = pl.read_csv(CASES_DIR / f"{cases_name}.csv")
    names = list(INPUTS)
    unit = np.stack([_unit(cases[name].to_numpy(), name) for name in names], axis=1)

    plt.figure(figsize=(12, 6))
    plt.imshow(unit, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Fraction of input range")
    plt.xticks(range(len(names)), [_short_label(n) for n in names])
    plt.ylabel("Case Index")
    plt.title(f"CFD case inputs ({cases_name}, {len(cases)} cases)")
    plt.savefig(PLOTS_DIR / f"{cases_name}_heatmap.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=(12, 6))
    for name in names:
        ax.plot(cases[name].to_numpy(), label=_short_label(name))
    ax.set_xlabel("Case Index")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Inputs across cases")
    ax.legend(ncol=2, fontsize=7)
    plt.savefig(PLOTS_DIR / f"{cases_name}_lineplot.png", dpi=300)
    plt.show()

    plot_pair_contour(cases, "aoa_deg", "sideslip_deg")
    plot_pair_grid(cases)


if __name__ == "__main__":
    main()
