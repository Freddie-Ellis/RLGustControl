"""Small helpers for writing OpenFOAM input files, shared by the mesh and solve stages."""

from collections.abc import Sequence
from pathlib import Path

import torch


def foam_vector(v: Sequence[float] | torch.Tensor) -> str:
    """Format a point or vector as OpenFOAM writes it, e.g. (0.1 0 -0.02)."""
    values = v.tolist() if isinstance(v, torch.Tensor) else v
    return "(" + " ".join(f"{x:.6g}" for x in values) + ")"


def write_foam_file(case_dir: Path, relative_path: str, body: str, foam_class: str = "dictionary") -> None:
    """Write an OpenFOAM input file with the standard FoamFile header (`foam_class` e.g. volVectorField)."""
    path = case_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {foam_class};\n    object      {path.name};\n}}\n\n"
    )
    path.write_text(header + body.strip() + "\n")
