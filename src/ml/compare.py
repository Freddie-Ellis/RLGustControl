"""Comparing saved runs: aligned training histories, a summary per run, and the config settings that differ.

No Qt here; the Compare tab only draws what this returns. Everything is read from the run directories (runs.py), so
it works for any run: train.py runs, distributed learner runs, and learner runs synced from the HPC.

Runs log at different rates in different units (train.py: one row per iteration; the distributed learner: one row per
episode plus one per `distributed.log_every` SAC updates), so histories share one x-axis: transitions, the data collected so far.
Every logged point is drawn as it is: nothing is smoothed, averaged or thinned out.
"""

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from src.ml.runs import RunSummary

X_LABEL = "transitions"  # data collected so far: one per control step, over all parallel envs
X_COLUMNS = ("transitions", "env_steps")  # the column that gives it: train.py logs the first, the learner the second
DEFAULT_METRICS = ("train_return", "eval_return", "critic_loss")
_NOT_METRICS = {"iteration", *X_COLUMNS}


@dataclass(frozen=True)
class Curve:
    """One run's history of one metric: every row that has a value for it, in order of data collected."""

    x: np.ndarray
    y: np.ndarray


def metrics_in(metrics: pl.DataFrame) -> list[str]:
    """The metric columns of one run's history (not its x-axis or row-key columns)."""
    return [c for c in metrics.columns if c not in _NOT_METRICS]


def metric_names(runs: list[RunSummary]) -> list[str]:
    """Every metric any of the runs logged, in a stable order."""
    seen: dict[str, None] = {}
    for s in runs:
        for column in metrics_in(s.run.read_metrics()):
            seen[column] = None
    return list(seen)


def x_column(metrics: pl.DataFrame) -> str | None:
    """The column that gives the transitions for this run, or None if it logged neither."""
    return next((c for c in X_COLUMNS if c in metrics.columns), None)


def curve(metrics: pl.DataFrame, metric: str) -> Curve | None:
    """`metric` against transitions, every row that has a value for it, where it happened in the run. Rows without
    a value are the other kind of row, not lost data: e.g. eval_return is empty on train episodes, losses on episode
    rows and before the first update. None if the run has no such metric, or no transitions column."""
    x_col = x_column(metrics)
    if x_col is None or metric not in metrics.columns:
        return None
    x = _spread_within_episodes(metrics[x_col].cast(pl.Float64).to_numpy())
    y = metrics[metric].cast(pl.Float64).to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any():
        return None
    order = np.argsort(x[keep], kind="stable")
    return Curve(x[keep][order], y[keep][order])


def _spread_within_episodes(x: np.ndarray) -> np.ndarray:
    """Rows logged at the same data collected, in logged order, spread evenly up to the next episode's data.

    A row's transitions is the data collected when it was logged. An episode row is the first at its value (the
    data grows when the episode comes in); the rows after it at the same value (SAC's metrics, every log_every
    updates) were logged while the learner worked through that data, before the next episode came in. So they
    happened between this value and the next: spread across that gap, in order, they show the run as it went,
    and never move into the wrong gap (no updates during warmup, say). After the last episode, the gap is the
    typical one between episodes. Rows alone at their value (every train.py row) are unchanged."""
    out = x.copy()
    finite = np.isfinite(x)
    values = np.unique(x[finite])
    if values.size == 0:
        return out
    typical_gap = float(np.median(np.diff(values))) if values.size > 1 else 1.0
    rows = np.flatnonzero(finite)
    start = 0
    while start < rows.size:
        end = start
        while end + 1 < rows.size and x[rows[end + 1]] == x[rows[start]]:
            end += 1
        if end > start:
            here = x[rows[start]]
            k = int(np.searchsorted(values, here, side="right"))
            gap = (values[k] - here) if k < values.size else typical_gap
            n = end - start + 1
            out[rows[start:end + 1]] = here + gap * np.arange(n) / n
        start = end + 1
    return out


# ----- summary -----
def _finite(metrics: pl.DataFrame, column: str) -> np.ndarray:
    """The column's finite values in logged order; empty if the run did not log it."""
    if column not in metrics.columns:
        return np.empty(0)
    values = metrics[column].cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def _last(values: np.ndarray) -> float | None:
    return float(values[-1]) if values.size else None


def _best(values: np.ndarray) -> float | None:
    return float(values.max()) if values.size else None


def _duration(s: RunSummary) -> timedelta | None:
    if s.status.finished is None:
        return None
    return s.status.finished.replace(tzinfo=None) - s.status.started.replace(tzinfo=None)


def summary(runs: list[RunSummary]) -> pl.DataFrame:
    """One row per run: what it is, how far it got, and its returns. Missing values are null."""
    rows: list[dict[str, Any]] = []
    for s in runs:
        m = s.run.read_metrics()
        x_col = x_column(m)
        evals, trains = _finite(m, "eval_return"), _finite(m, "train_return")
        duration = _duration(s)
        rows.append({
            "run": s.run.name,
            "status": s.status.state,
            "config": s.config.describe(),
            "transitions": _last(_finite(m, x_col)) if x_col else None,
            "rows logged": m.height,
            "best eval return": _best(evals),
            "final eval return": _last(evals),
            "best train return": _best(trains),
            "final train return": _last(trains),
            "duration [min]": duration.total_seconds() / 60 if duration else None,
        })
    return pl.DataFrame(rows, schema_overrides={k: pl.Float64 for k in (
        "transitions", "best eval return", "final eval return", "best train return", "final train return",
        "duration [min]",
    )})


# ----- config differences -----
def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out |= _flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    return {prefix: value}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def config_differences(runs: list[RunSummary]) -> pl.DataFrame:
    """The settings that differ between the runs: one row per dotted setting (e.g. `algo.lr`), one column per run.
    A setting a run does not have (another env kind, say) shows as empty. Empty frame if they all agree."""
    flat = [_flatten(s.config.model_dump(mode="json")) for s in runs]
    keys = list(dict.fromkeys(k for f in flat for k in f))
    missing = object()
    differing = [
        k for k in keys
        if any(not _same(f.get(k, missing), flat[0].get(k, missing)) for f in flat[1:])
    ]
    columns: dict[str, list[str]] = {"setting": differing}
    names = _unique([s.run.name for s in runs])
    for name, f in zip(names, flat):
        columns[name] = ["" if k not in f else str(f[k]) for k in differing]
    return pl.DataFrame(columns, schema={c: pl.String for c in columns})


def _unique(names: list[str]) -> list[str]:
    """Column names must be unique; two runs with the same folder name (from different roots) get a suffix."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out
