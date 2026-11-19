"""Pieces shared by the training scripts of the sparse-tap latent networks (train_ae.py, train_regressor.py)."""

import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


@dataclass(frozen=True)
class Standardiser:
    """Per-column shift and scale fitted on the training runs."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x: Tensor) -> Standardiser:
        return cls(mean=x.mean(dim=0), std=x.std(dim=0).clamp_min(1e-8))

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean


def split_rows(n: int, val_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    """(training, validation) row indices, chosen at random."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = round(val_fraction * n)
    return order[n_val:].sort().values, order[:n_val].sort().values


def r_squared(pred: Tensor, true: Tensor) -> Tensor:
    """Per column, 1 - residual / total sum of squares about the column's own mean. (K,)"""
    residual = (pred - true).pow(2).sum(dim=0)
    total = (true - true.mean(dim=0)).pow(2).sum(dim=0).clamp_min(1e-12)
    return 1.0 - residual / total


def new_run_dir(root: Path, name: str) -> Path:
    """root/<YYYYmmdd-HHMMSS>_<name>, created."""
    run_dir = root / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    run_dir.mkdir(parents=True)
    return run_dir


class TomlConfig(BaseModel):
    """A training run's settings: read from a TOML file in configs/, saved as the run's config.json."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        """Settings the file leaves out keep their defaults."""
        with open(path, "rb") as f:
            return cls.model_validate(tomllib.load(f))


def config_path(default: Path) -> Path:
    """The TOML config a training script was given (`python -m <script> <config.toml>`), else `default`."""
    return Path(sys.argv[1]) if len(sys.argv) > 1 else default
