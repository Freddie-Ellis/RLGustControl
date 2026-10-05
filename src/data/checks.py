"""Generic integrity checks for time-series datasets. Extend with dataset-specific checks once formats are known."""

from typing import Literal

import polars as pl
from pydantic import BaseModel

type Level = Literal["ok", "info", "warning", "error"]

UNIFORM_DT_TOLERANCE = 0.01  # relative spread of time steps allowed before warning


class Finding(BaseModel):
    level: Level
    message: str


def check_table(df: pl.DataFrame, time_col: str | None = None) -> list[Finding]:
    if df.height == 0:
        return [Finding(level="error", message="Table is empty")]

    out = [Finding(level="info", message=f"{df.height} rows x {df.width} columns")]
    numeric = [c for c, dt in df.schema.items() if dt.is_numeric()]
    other = [c for c in df.columns if c not in numeric]
    if other:
        out.append(Finding(level="info", message=f"Non-numeric columns: {', '.join(other)}"))

    # Missing values: nulls anywhere, NaNs in float columns.
    nulls = df.null_count().row(0, named=True)
    floats = [c for c in numeric if df.schema[c].is_float()]
    nans = df.select(pl.col(floats).is_nan().sum()).row(0, named=True) if floats else {}
    missing = {c: nulls[c] + nans.get(c, 0) for c in df.columns}
    bad = {c: n for c, n in missing.items() if n}
    if bad:
        out.append(Finding(level="warning", message="Missing values: " + ", ".join(f"{c} ({n})" for c, n in bad.items())))
    else:
        out.append(Finding(level="ok", message="No missing values"))

    # Constant columns often mean a dead or disconnected sensor.
    if numeric:
        n_unique = df.select(pl.col(numeric).n_unique()).row(0, named=True)
        constant = [c for c, n in n_unique.items() if n <= 1]
        if constant:
            out.append(Finding(level="warning", message=f"Constant columns (dead sensor?): {', '.join(constant)}"))

    if time_col is not None:
        out.extend(_check_time(df, time_col))
    return out


def _check_time(df: pl.DataFrame, time_col: str) -> list[Finding]:
    if not df.schema[time_col].is_numeric():
        return [Finding(level="error", message=f"Time column '{time_col}' is not numeric")]
    dt = df.select(pl.col(time_col).diff().drop_nulls()).to_series()
    if dt.len() == 0:
        return [Finding(level="warning", message="Only one sample: cannot check time steps")]
    out: list[Finding] = []
    n_bad = int((dt <= 0).sum())
    if n_bad:
        out.append(Finding(level="error", message=f"Time is not strictly increasing ({n_bad} steps <= 0)"))
    median = float(dt.median() or 0.0)
    if median > 0:
        spread = (float(dt.max() or 0.0) - float(dt.min() or 0.0)) / median
        rate = f"median dt {median:.6g} ({1 / median:.6g} Hz)"
        if spread > UNIFORM_DT_TOLERANCE:
            out.append(Finding(level="warning", message=f"Non-uniform sampling: {rate}, dt range [{dt.min():.6g}, {dt.max():.6g}]"))
        else:
            out.append(Finding(level="ok", message=f"Uniform sampling: {rate}"))
    return out
