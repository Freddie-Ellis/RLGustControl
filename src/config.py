from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)   # read-only after creation

    # paths
    runs_dir: Path = ROOT / "runs"
    configs_dir: Path = ROOT / "configs"

    # compute
    device: Literal["cpu", "cuda"] = "cpu"
    default_seed: int = 0

    # training defaults
    gamma: float = 0.98
    lr: float = 3e-4
    n_envs: int = 64

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


CONFIG = Config()