from pathlib import Path

import polars as pl

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".ipc", ".arrow", ".feather")


def load_table(path: Path) -> pl.DataFrame:
    """Load a tabular dataset (sensor / CFD / experiment) into Polars, chosen by file extension."""
    match path.suffix.lower():
        case ".csv":
            return pl.read_csv(path, infer_schema_length=10_000)
        case ".parquet":
            return pl.read_parquet(path)
        case ".ipc" | ".arrow" | ".feather":
            return pl.read_ipc(path)
        case other:
            raise ValueError(f"unsupported file type {other!r}; expected one of {SUPPORTED_SUFFIXES}")
