"""Find the slice images rendered by paraview_scripts/slices.py, for flicking through them.

slices.py writes, per case and per slice field:

    <images dir>/<batch>_<case>/<field>_sweep_<axis>_<index>.png

with x slices numbered upstream to downstream and y slices port to starboard. Every image shares one camera and the
model stays still (model frame), so stepping through a sweep looks like sliding a cut plane through the flow.
"""

import re
from pathlib import Path

SLICE_IMAGES_DIR: Path = Path(__file__).resolve().parent.parent / "db" / "cfd"  # slices.py's <outdir>, copied off the HPC
AXES: tuple[str, ...] = ("x", "y")

_SWEEP = re.compile(r"^(?P<field>.+)_sweep_(?P<axis>[a-z])_(?P<index>\d+)\.png$")


def list_cases(images_dir: Path = SLICE_IMAGES_DIR) -> list[str]:
    """Case folders under `images_dir` that hold at least one sweep image, sorted by name."""
    if not images_dir.is_dir():
        return []
    return sorted(d.name for d in images_dir.iterdir() if d.is_dir() and any(_sweep_images(d)))


def list_fields(case_dir: Path) -> list[str]:
    """Slice fields rendered for this case (e.g. Cp, U, k), sorted by name."""
    return sorted({m["field"] for m in _sweep_images(case_dir)})


def sweep(case_dir: Path, field: str, axis: str) -> list[Path]:
    """One field's images along one axis, in slice order. Empty if that sweep was not rendered."""
    found = [(int(m["index"]), case_dir / m.string) for m in _sweep_images(case_dir)
             if m["field"] == field and m["axis"] == axis]
    return [path for _, path in sorted(found)]


def _sweep_images(case_dir: Path) -> list[re.Match[str]]:
    return [m for p in case_dir.glob("*_sweep_*.png") if (m := _SWEEP.match(p.name))]
