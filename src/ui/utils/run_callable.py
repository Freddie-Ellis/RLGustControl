"""Child side of TelemetryConsole.run: load a pickled (function, args, kwargs) and call it.

    python -u -m src.ui.utils.run_callable <pickle file>

The function is pickled by reference (module + name), so it must be importable at top level: no lambdas or closures.
The file is deleted once read. An uncaught exception prints its traceback to stderr and exits with code 1, as usual.
No Qt here: this runs in the learner/actor process.
"""

import pickle
import sys
from collections.abc import Callable
from pathlib import Path

type Job = tuple[Callable[..., object], tuple[object, ...], dict[str, object]]


def main(path: Path) -> None:
    with path.open("rb") as f:
        fn, args, kwargs = pickle.load(f)
    path.unlink(missing_ok=True)
    fn(*args, **kwargs)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
