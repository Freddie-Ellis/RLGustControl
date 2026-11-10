from functools import cache
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent


@cache
def _cuda_available() -> bool:
    """Asked once, on first use: the CUDA device query takes over a second, too slow to pay on every import."""
    return torch.cuda.is_available()


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)   # read-only after creation

    # paths
    runs_dir: Path = ROOT / "runs"
    configs_dir: Path = ROOT / "configs"
    profiles_dir: Path = ROOT / "profiles"

    # compute
    device: Literal["auto", "cpu", "cuda"] = "auto"  # auto = cuda if available

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if _cuda_available() else "cpu")
        return torch.device(self.device)


CONFIG = Config()
