"""Control authority and the largest gust the surfaces can correct, from the dataset written by src/cfd/dataset.py
(setpoints.parquet).

    .venv/Scripts/python.exe -m postpro.authority data/Batch2/setpoints.parquet [--out FOLDER] [--show]

Writes to <out> (default <parquet folder>/postpro/authority/):
    additivity_combined.png
        change in C_L, C_D and C_m from the clean wing with every surface type deflected at once (the combined case),
        against the sum of the changes from each surface type deflected alone at the same angle, beta 0. One point
        per alpha. On the dashed line the effects add exactly
    additivity_sides.png
        change from both sides deflected, against twice the change from the right side only at the same deflection
        and alpha, beta 0. One colour per surface type and deflection, one symbol per alpha, one legend entry per
        point
    authority.png
        change from the clean wing against alpha at beta 0, one colour per surface type and one symbol per
        deflection. Top: C_L and C_m with both sides deflected, and the combined case. Bottom: C_l and C_n with the
        right side deflected (the left side gives the opposite sign)
    lift_and_pitch.png
        one panel per alpha: each surface type's (ΔC_L, ΔC_m) at each deflection, joined through the clean wing
        (0, 0), and the clean wing's values at each lower alpha (black, labelled with that alpha) relative to the
        clean wing at this alpha. After an upward gust from a lower alpha to this one, a surface holds both lift and
        pitch where its line passes through that alpha's black point
    lift_position.png
        the same question shown another way: where along the model's x axis each surface type's change in lift acts,
        as a fraction of the root chord from the nose (x = x_ref - c ΔC_m / ΔC_L; 0 is the nose, 1 the centreline
        trailing edge), against alpha at beta 0, one symbol per deflection (LE flaps left out), and where a
        small gust's extra lift acts on the clean wing (black, from the slopes of C_L and C_m against alpha; left
        out past the stall, where a gust adds no lift). A surface holds lift and pitch together only where its
        points are on the black line. The gap between them hardly depends on the C_m reference point (dashed grey)
    gust_upward_crossing.png, gust_side_crossing.png
        how max_gust.png is found. One panel per angle before the gust (columns) and coefficient (rows): the change
        needed against gust speed (black; crosses where the clean wing was solved) and the largest decrease and
        increase each surface type reaches (dots where it was solved). A surface corrects the gust until the black
        line leaves the range between its two lines. The top axis gives the angle after the gust
    max_gust.png
        the largest gust that can be corrected. Top: upward gust against alpha before it, holding C_L or C_m, with
        both sides deflected. Bottom: side gust from beta 0 against alpha, holding C_l or C_n, with one side
        deflected (either side, so either sign). Every smaller gust is corrected too. The grey line is where the
        gust reaches the last angle solved, so points on it are lower limits

How the largest gust is found: a gust w (upward) or v (sideways) increases alpha or beta by atan(w / V). After it,
the change the surfaces must make is the clean wing's coefficient at the angle before minus its value at the angle
after. That change can be made when it lies between the largest decrease and increase the deflections reach at the
angle after (any value between 0 and a solved deflection is taken to be reachable at a deflection in between). The
gust is increased in steps of GUST_STEP_DEG until it can't. Mean values, interpolated linearly in the angle.

Error bars are worst-case bounds from the min and max over the averaging window (see aircraft_changes in
postpro/surfaces.py), added together for sums; hollow markers mean a set point involved is unconverged. C_m is about
the reference point in setpoints.parquet (ref_x_m, ref_y_m, ref_z_m).
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from postpro.surfaces import AIRCRAFT_COEFFICIENTS, ALPHA_MARKERS, FAMILIES, aircraft_changes, finish, plot_line, save
from src.cfd.dataset import SURFACE_FAMILIES

BOUNDS: tuple[str, ...] = ("", "_lo", "_hi")  # d<C>, and its lower and upper bound
LONGITUDINAL: tuple[str, ...] = ("C_L", "C_m")  # held in an upward gust, by deflecting both sides
LATERAL: tuple[str, ...] = ("C_l", "C_n")  # held in a side gust, by deflecting one side
CONFIGURATIONS: tuple[str, ...] = (*FAMILIES, "combined")
DEFLECTION_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")  # a surface type's deflections, smallest first
GUST_STEP_DEG = 0.1  # resolution of the largest gust angle
# Centreline nose position and root chord (m), estimated from the plan view (see the comment above [reference] in
# cad/uav/geometry.toml); the dataset doesn't store them.
NOSE_X_M = -0.020
ROOT_CHORD_M = 0.520


def colour(family: str) -> str:
    """The same colour for a surface type (or the combined case) on every plot."""
    return f"C{CONFIGURATIONS.index(family)}"


def combined_deflections(df: pl.DataFrame) -> dict[str, float]:
    """Each surface type's deflection in the combined case, leaving out those not deflected."""
    row = df.filter(pl.col("mode") == "combined").row(0, named=True)
    first = {family: surface for surface, family in reversed(SURFACE_FAMILIES.items())}  # one surface per type
    angles = {family: row[f"{first[family]}_r_deg"] for family in FAMILIES}
    return {family: delta for family, delta in angles.items() if delta != 0}


def label(df: pl.DataFrame, family: str) -> str:
    if family != "combined":
        return family
    return "all at once (" + ", ".join(f"{f} {d:g}°" for f, d in combined_deflections(df).items()) + ")"


def clean_level(df: pl.DataFrame) -> pl.DataFrame:
    """The clean wing at beta 0, by alpha."""
    return df.filter(pl.col("mode") == "clean", pl.col("beta_deg") == 0).sort("alpha_deg")


def both_sides(changes: pl.DataFrame) -> pl.DataFrame:
    """Changes at beta 0 with both sides deflected, and the combined case."""
    return changes.filter(pl.col("beta_deg") == 0, pl.col("mode").is_in(["sym", "combined"]))


def gust_speed(speed: float, angle: Any) -> Any:
    """Gust speed (m/s) that turns the flow by `angle` (deg)."""
    return speed * np.tan(np.radians(angle))


def gust_angle(speed: float, gust: Any) -> Any:
    """Change in flow angle (deg) from a gust (m/s)."""
    return np.degrees(np.arctan(gust / speed))


# Additivity

def combined_vs_sum(df: pl.DataFrame, changes: pl.DataFrame) -> pl.DataFrame:
    """The combined case's d<C> at beta 0, joined on alpha to the sum (suffix _sum) of the single surface types'
    d<C> at the same deflections."""
    deflections = combined_deflections(df)
    singles = [changes.filter(pl.col("family") == family, pl.col("mode") == "sym", pl.col("delta_deg") == delta,
                              pl.col("beta_deg") == 0) for family, delta in deflections.items()]
    summed = pl.concat(singles).group_by("alpha_deg").agg(
        pl.len().alias("surface_types"), pl.col("converged").all(),
        *(pl.col(f"d{c}{end}").sum() for c in AIRCRAFT_COEFFICIENTS for end in BOUNDS))
    combined = changes.filter(pl.col("mode") == "combined", pl.col("beta_deg") == 0)
    return (combined.join(summed.filter(pl.col("surface_types") == len(singles)), on="alpha_deg", suffix="_sum")
            .sort("alpha_deg"))


def sym_vs_right(changes: pl.DataFrame) -> pl.DataFrame:
    """Both-sides d<C> at beta 0, joined to twice the right-only d<C> (suffix _right) at the same surface type,
    deflection and alpha."""
    at_zero = changes.filter(pl.col("beta_deg") == 0)
    doubled = at_zero.filter(pl.col("mode") == "right").select(
        "family", "delta_deg", "alpha_deg", "converged",
        *((2 * pl.col(f"d{c}{end}")).alias(f"d{c}{end}") for c in AIRCRAFT_COEFFICIENTS for end in BOUNDS))
    return (at_zero.filter(pl.col("mode") == "sym").join(doubled, on=["family", "delta_deg", "alpha_deg"],
                                                         suffix="_right")
            .sort("family", "delta_deg", "alpha_deg"))


def parity_point(ax: plt.Axes, row: dict[str, Any], c: str, suffix: str, colour: str, marker: str,
                 label: str | None) -> None:
    """One point: d<C><suffix> across, d<C> up, with their bounds as error bars; hollow unless both converged."""
    x, y = row[f"d{c}{suffix}"], row[f"d{c}"]
    converged = row["converged"] and row[f"converged{suffix}"]
    ax.errorbar([x], [y], xerr=[[x - row[f"d{c}_lo{suffix}"]], [row[f"d{c}_hi{suffix}"] - x]],
                yerr=[[y - row[f"d{c}_lo"]], [row[f"d{c}_hi"] - y]], fmt="none", ecolor=colour, capsize=2, lw=1)
    ax.plot([x], [y], marker, color=colour, mfc=colour if converged else "white", ms=6, label=label)


def diagonal(ax: plt.Axes) -> None:
    """The line y = x across the plot, with equal axis ranges."""
    low = min(ax.get_xlim()[0], ax.get_ylim()[0])
    high = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def plot_additivity_combined(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    rows = combined_vs_sum(df, changes)
    fig, axes = plt.subplots(1, len(AIRCRAFT_COEFFICIENTS), figsize=(18, 5.5))
    for i, row in enumerate(rows.iter_rows(named=True)):
        for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
            parity_point(ax, row, c, "_sum", f"C{i}", "o", f"α {row['alpha_deg']:g}°")
    for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
        diagonal(ax)
        finish(ax, df, f"sum of single surface types, Δ{c}", c, f"all together, Δ{c}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label(df, 'combined')} against the sum of single surface types, β = 0°")
    save(fig, path, show)


def plot_additivity_sides(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    rows = sym_vs_right(changes)
    alphas = sorted(rows["alpha_deg"].unique().to_list())
    pairs = rows.select("family", "delta_deg").unique(maintain_order=True).rows()
    fig, axes = plt.subplots(1, len(AIRCRAFT_COEFFICIENTS), figsize=(18, 5.5))
    for row in rows.iter_rows(named=True):
        pair = (row["family"], row["delta_deg"])
        for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
            parity_point(ax, row, c, "_right", f"C{pairs.index(pair) % 10}",
                         ALPHA_MARKERS[alphas.index(row["alpha_deg"])],
                         f"{pair[0]} {pair[1]:g}°, α {row['alpha_deg']:g}°")
    for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
        diagonal(ax)
        finish(ax, df, f"2 × right side only, Δ{c}", c, f"both sides, Δ{c}")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Both sides against twice one side, β = 0°")
    save(fig, path, show)


# Authority

def deflection_lines(df: pl.DataFrame, rows: pl.DataFrame,
                     families: tuple[str, ...]) -> Iterator[tuple[str, str, str, pl.DataFrame]]:
    """For each surface type and deflection in `rows`: its colour, symbol, legend label, and rows by alpha."""
    for family in families:
        sub = rows.filter(pl.col("family") == family)
        for marker, delta in zip(DEFLECTION_MARKERS, sub["delta_deg"].unique().sort().to_list()):
            name = label(df, family) if delta is None else f"{family} {delta:g}°"  # the combined case: None
            yield colour(family), marker, name, sub.filter(pl.col("delta_deg").eq_missing(delta)).sort("alpha_deg")


def plot_deflections(ax: plt.Axes, df: pl.DataFrame, rows: pl.DataFrame, c: str, families: tuple[str, ...]) -> None:
    """d<c> against alpha, one line per surface type (colour) and deflection (symbol)."""
    for line_colour, marker, name, line in deflection_lines(df, rows, families):
        x, y = line["alpha_deg"].to_numpy(), line[f"d{c}"].to_numpy()
        ax.errorbar(x, y, yerr=[y - line[f"d{c}_lo"].to_numpy(), line[f"d{c}_hi"].to_numpy() - y], fmt="none",
                    ecolor=line_colour, capsize=2, lw=1)
        plot_line(ax, x, y, line["converged"].to_numpy(), line_colour, marker=marker, label=name)
    ax.axhline(0, color="grey", lw=0.5)
    finish(ax, df, "α (deg)", c, f"Δ{c}")


def plot_authority(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    right = changes.filter(pl.col("beta_deg") == 0, pl.col("mode") == "right")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, c in zip(axes[0], LONGITUDINAL):
        plot_deflections(ax, df, both_sides(changes), c, CONFIGURATIONS)
        ax.set_title("both sides deflected")
    for ax, c in zip(axes[1], LATERAL):
        plot_deflections(ax, df, right, c, FAMILIES)
        ax.set_title("right side deflected (left: opposite sign)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Change from the clean wing, β = 0°")
    save(fig, path, show)


# Lift and pitch together

def plot_lift_and_pitch(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    rows = both_sides(changes)
    clean = clean_level(df)
    alphas = sorted(rows["alpha_deg"].unique().to_list())
    columns = 3
    fig, axes = plt.subplots(-(-len(alphas) // columns), columns, figsize=(18, 10), squeeze=False)
    for ax, alpha in zip(axes.flat, alphas):
        for family in CONFIGURATIONS:
            sub = rows.filter(pl.col("family") == family, pl.col("alpha_deg") == alpha)
            # The clean wing is (0, 0); sort it in with the deflections (the combined case has no single deflection).
            order = np.argsort(np.append(sub["delta_deg"].fill_null(1.0).to_numpy(), 0.0))
            x, y = (np.append(sub[f"d{c}"].to_numpy(), 0.0)[order] for c in LONGITUDINAL)
            plot_line(ax, x, y, np.append(sub["converged"].to_numpy(), True)[order], colour(family),
                      label=label(df, family))
        before = clean.filter(pl.col("alpha_deg") < alpha)
        now = clean.filter(pl.col("alpha_deg") == alpha)
        x, y = ((before[f"total__{c}_mean"] - now[f"total__{c}_mean"][0]).to_numpy() for c in LONGITUDINAL)
        ax.plot(x, y, "k--x", lw=1, label="clean wing at a lower α")
        for a, xi, yi in zip(before["alpha_deg"].to_list(), x, y):
            ax.annotate(f"{a:g}°", (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"α {alpha:g}°")
        ax.set_xlabel("ΔC_L")
        ax.set_ylabel("ΔC_m")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Change in lift and pitch from the clean wing, β = 0°, C_m about x = {df['ref_x_m'][0]:.3f} m")
    save(fig, path, show)


def plot_lift_position(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    ref_x, chord = float(df["ref_x_m"][0]), float(df["ref_chord_m"][0])

    def along_root(moment_per_lift: Any) -> Any:
        """Fraction of the root chord from the nose at which a change in lift with this ΔC_m / ΔC_L acts."""
        return (ref_x - chord * moment_per_lift - NOSE_X_M) / ROOT_CHORD_M

    # Past the trailing edge (> 1) the change is mostly a pitching moment with little net lift, so the point is cut.
    x_lift = along_root(pl.col("dC_m") / pl.col("dC_L"))
    rows = both_sides(changes).with_columns(pl.when(x_lift <= 1).then(x_lift).alias("x_lift"))
    fig, ax = plt.subplots(figsize=(11, 7))
    for line_colour, marker, name, line in deflection_lines(df, rows, tuple(f for f in CONFIGURATIONS if f != "le")):
        plot_line(ax, line["alpha_deg"].to_numpy(), line["x_lift"].to_numpy(), line["converged"].to_numpy(),
                  line_colour, marker=marker, label=name)
    clean = clean_level(df)
    alpha = clean["alpha_deg"].to_numpy()
    lift_slope = np.gradient(clean["total__C_L_mean"].to_numpy(), alpha)
    pitch_slope = np.gradient(clean["total__C_m_mean"].to_numpy(), alpha)
    ax.plot(alpha, np.where(lift_slope > 0, along_root(pitch_slope / lift_slope), np.nan), "k-x", lw=2,
            label="clean wing, small gust")
    ax.axhline(along_root(0.0), color="grey", ls="--", lw=1, label="C_m reference point")
    ax.set_ylim(0, 1)
    ax.set_xlabel("α (deg)")
    ax.set_ylabel("where the change in lift acts, fraction of root chord from the nose")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Where the change in lift acts, β = 0°")
    save(fig, path, show)


# Gusts

def reach(rows: pl.DataFrame, c: str, by: str) -> pl.DataFrame:
    """For each surface type (family) and value of the column `by` (renamed "angle"), the largest decrease (low)
    and increase (high) in d<c> among `rows`, counting the undeflected wing (0)."""
    return (rows.group_by("family", pl.col(by).alias("angle"))
            .agg(pl.min_horizontal(pl.col(f"d{c}").min(), pl.lit(0.0)).alias("low"),
                 pl.max_horizontal(pl.col(f"d{c}").max(), pl.lit(0.0)).alias("high"))
            .sort("family", "angle"))


def one_side_changes(df: pl.DataFrame, c: str) -> pl.DataFrame:
    """d<c> (C_l or C_n) with one side deflected, at beta 0 and above: the right-side runs, and the left side, which
    is the right-side run at -beta mirrored (C_l and C_n change sign). Against the clean wing at the same alpha and
    beta."""
    right = df.filter(pl.col("mode") == "right").select("family", "alpha_deg", "beta_deg",
                                                        pl.col(f"total__{c}_mean").alias("value"))
    left = right.with_columns(-pl.col("beta_deg"), -pl.col("value"))
    clean = df.filter(pl.col("mode") == "clean").select("alpha_deg", "beta_deg",
                                                        pl.col(f"total__{c}_mean").alias("clean"))
    return (pl.concat([right, left]).filter(pl.col("beta_deg") >= 0).join(clean, on=["alpha_deg", "beta_deg"])
            .with_columns((pl.col("value") - pl.col("clean")).alias(f"d{c}")))


def upward(df: pl.DataFrame, changes: pl.DataFrame, c: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """For an upward gust: the clean wing's alpha and <c> at beta 0, and the reach of both sides deflected."""
    clean = clean_level(df)
    return (clean["alpha_deg"].to_numpy(), clean[f"total__{c}_mean"].to_numpy(),
            reach(both_sides(changes), c, "alpha_deg"))


def sideways(df: pl.DataFrame, c: str, alpha: float) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """For a side gust at `alpha`: the clean wing's beta and <c>, and the reach of one side deflected."""
    clean = df.filter(pl.col("mode") == "clean", pl.col("alpha_deg") == alpha).sort("beta_deg")
    return (clean["beta_deg"].to_numpy(), clean[f"total__{c}_mean"].to_numpy(),
            reach(one_side_changes(df, c).filter(pl.col("alpha_deg") == alpha), c, "beta_deg"))


def needed_after(start: float, clean_x: np.ndarray, clean_value: np.ndarray,
                 table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flow angles after a gust from `start`, in GUST_STEP_DEG steps up to the last angle solved, and the change
    the surfaces must make at each to get back to the clean wing's value at `start`."""
    last = min(float(clean_x[-1]), float(table["angle"].to_numpy().max()))
    after = np.arange(start, last + GUST_STEP_DEG / 2, GUST_STEP_DEG)
    return after, np.interp(start, clean_x, clean_value) - np.interp(after, clean_x, clean_value)


def max_gust_angle(start: float, clean_x: np.ndarray, clean_value: np.ndarray, reached: pl.DataFrame) -> float:
    """The largest increase in the flow angle (deg) from `start` for which every smaller one can be corrected by
    one surface type (`reached`, from reach), up to the last angle solved."""
    after, needed = needed_after(start, clean_x, clean_value, reached)
    x = reached["angle"].to_numpy()
    ok = ((np.interp(after, x, reached["low"].to_numpy()) <= needed)
          & (needed <= np.interp(after, x, reached["high"].to_numpy())))
    fails = np.flatnonzero(~ok)
    return float(after[max(fails[0] - 1, 0)] if fails.size else after[-1]) - start


def plot_crossing(ax: plt.Axes, df: pl.DataFrame, start: float, clean_x: np.ndarray, clean_value: np.ndarray,
                  table: pl.DataFrame, families: tuple[str, ...], angle: str) -> None:
    """The change needed after a gust from `start` against gust speed (black), and each surface type's largest
    decrease and increase (from reach), up to the last angle solved."""
    speed = float(df["speed_m_s"][0])
    after, needed = needed_after(start, clean_x, clean_value, table)
    for family in families:
        sub = table.filter(pl.col("family") == family)
        x = sub["angle"].to_numpy()
        solved = (x >= start) & (x <= after[-1] + GUST_STEP_DEG / 2)
        for end in ("low", "high"):
            y = sub[end].to_numpy()
            ax.plot(gust_speed(speed, after - start), np.interp(after, x, y), color=colour(family), lw=1,
                    label=label(df, family) if end == "low" else None)
            ax.plot(gust_speed(speed, x[solved] - start), y[solved], "o", color=colour(family), ms=4)
    solved = (clean_x >= start) & (clean_x <= after[-1] + GUST_STEP_DEG / 2)
    ax.plot(gust_speed(speed, after - start), needed, "k-", lw=2, label="needed")
    ax.plot(gust_speed(speed, clean_x[solved] - start), np.interp(start, clean_x, clean_value) - clean_value[solved],
            "kx", ms=7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3)
    ax.secondary_xaxis("top", functions=(lambda w: start + gust_angle(speed, w),
                                         lambda a: gust_speed(speed, a - start))).set_xlabel(
        f"{angle} after gust (deg)")


def plot_gust_upward(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    starts = sorted(both_sides(changes)["alpha_deg"].unique().to_list())[:-1]  # the last alpha has nowhere to go
    fig, axes = plt.subplots(len(LONGITUDINAL), len(starts), figsize=(4.5 * len(starts), 10), squeeze=False)
    for row, c in zip(axes, LONGITUDINAL):
        alpha, value, table = upward(df, changes, c)
        for ax, start in zip(row, starts):
            plot_crossing(ax, df, start, alpha, value, table, CONFIGURATIONS, "α")
            ax.set_title(f"α before {start:g}°")
            ax.set_xlabel("upward gust, w (m/s)")
            ax.set_ylabel(f"Δ{c}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Upward gust: change needed and reached, both sides deflected, β = 0°")
    save(fig, path, show)


def plot_gust_side(df: pl.DataFrame, path: Path, show: bool) -> None:
    alphas = sorted(df.filter(pl.col("mode") == "right")["alpha_deg"].unique().to_list())
    fig, axes = plt.subplots(len(LATERAL), len(alphas), figsize=(4.5 * len(alphas), 10), squeeze=False)
    for row, c in zip(axes, LATERAL):
        for ax, alpha in zip(row, alphas):
            plot_crossing(ax, df, 0.0, *sideways(df, c, alpha), FAMILIES, "β")
            ax.set_title(f"α {alpha:g}°")
            ax.set_xlabel("side gust, v (m/s)")
            ax.set_ylabel(f"Δ{c}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Side gust from β 0°: change needed and reached, one side deflected")
    save(fig, path, show)


def gust_axes(ax: plt.Axes, speed: float, title: str, xlabel: str, gust: str, angle: str) -> None:
    """Labels, grid and a right-hand axis giving the gust as a change in flow angle."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"largest gust, {gust} (m/s)")
    ax.grid(alpha=0.3)
    ax.secondary_yaxis("right", functions=(lambda w: gust_angle(speed, w),
                                           lambda d: gust_speed(speed, d))).set_ylabel(f"{angle} (deg)")


def plot_max_gust(df: pl.DataFrame, changes: pl.DataFrame, path: Path, show: bool) -> None:
    speed = float(df["speed_m_s"][0])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, c in zip(axes[0], LONGITUDINAL):
        alpha, value, table = upward(df, changes, c)
        last = min(float(alpha[-1]), float(table["angle"].to_numpy().max()))
        before = alpha[alpha < last]
        for family in CONFIGURATIONS:
            reached = table.filter(pl.col("family") == family)
            angles = np.array([max_gust_angle(a, alpha, value, reached) for a in before])
            ax.plot(before, gust_speed(speed, angles), "-o", color=colour(family), ms=4, lw=1, label=label(df, family))
        ax.plot(before, gust_speed(speed, last - before), "--", color="grey", lw=1,
                label=f"end of data (gust reaches α {last:g}°)")
        gust_axes(ax, speed, f"upward gust, holding {c}", "α before gust (deg)", "w", "Δα")
    alphas = sorted(df.filter(pl.col("mode") == "right")["alpha_deg"].unique().to_list())
    for ax, c in zip(axes[1], LATERAL):
        cases = [sideways(df, c, a) for a in alphas]
        for family in FAMILIES:
            angles = np.array([max_gust_angle(0.0, beta, value, table.filter(pl.col("family") == family))
                               for beta, value, table in cases])
            ax.plot(alphas, gust_speed(speed, angles), "-o", color=colour(family), ms=4, lw=1, label=family)
        last = float(cases[0][0][-1])
        ax.axhline(gust_speed(speed, last), ls="--", color="grey", lw=1, label=f"end of data (gust reaches β {last:g}°)")
        gust_axes(ax, speed, f"side gust from β 0°, holding {c}", "α (deg)", "v", "Δβ")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"Largest gust corrected at {speed:g} m/s")
    save(fig, path, show)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Control authority from setpoints.parquet.")
    parser.add_argument("setpoints", type=Path, help="setpoints.parquet from src/cfd/dataset.py")
    parser.add_argument("--out", type=Path, help="output folder (default: <parquet folder>/postpro/authority)")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    args = parser.parse_args(argv)

    df = pl.read_parquet(args.setpoints)
    out: Path = args.out or args.setpoints.parent / "postpro" / "authority"
    changes = aircraft_changes(df, (*AIRCRAFT_COEFFICIENTS, *LATERAL))

    plot_additivity_combined(df, changes, out / "additivity_combined.png", args.show)
    plot_additivity_sides(df, changes, out / "additivity_sides.png", args.show)
    plot_authority(df, changes, out / "authority.png", args.show)
    plot_lift_and_pitch(df, changes, out / "lift_and_pitch.png", args.show)
    plot_lift_position(df, changes, out / "lift_position.png", args.show)
    plot_gust_upward(df, changes, out / "gust_upward_crossing.png", args.show)
    plot_gust_side(df, out / "gust_side_crossing.png", args.show)
    plot_max_gust(df, changes, out / "max_gust.png", args.show)
    print(f"Written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
