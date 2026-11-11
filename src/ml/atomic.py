"""Writing files that another process (the UI, the sync, the actor) may read at any moment.

Write to `<name>.tmp`, then rename over the real name, so a reader sees the old file or the new one, never half of
one. On Windows the rename is refused while anything has the target open, even for the few milliseconds a reader
takes, so it is tried a few times.
"""

import time
from pathlib import Path

REPLACE_TRIES = 10
REPLACE_WAIT_S = 0.05  # between tries: a reader holds the file for milliseconds


def tmp_path(path: Path) -> Path:
    """`ep_1.json` -> `ep_1.json.tmp`. Keeping the full name stops ep_1.json and ep_1.parquet sharing a tmp file."""
    return path.with_name(path.name + ".tmp")


def replace(tmp: Path, path: Path) -> None:
    """Rename `tmp` over `path`. Raises PermissionError if `path` stayed open elsewhere through every try."""
    for attempt in range(REPLACE_TRIES):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = tmp_path(path)
    tmp.write_text(text)
    replace(tmp, path)
