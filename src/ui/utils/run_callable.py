"""Child side of TelemetryConsole.run: load a pickled (function, args, kwargs) and call it.

    python -u -m src.ui.utils.run_callable <pickle file> [<profile output .prof>]

The function is pickled by reference (module + name), so it must be importable at top level: no lambdas or closures.
The file is deleted once read. An uncaught exception prints its traceback to stderr and exits with code 1, as usual.
With a profile path, the call runs under cProfile and the stats are written when it returns or raises (not if the
process is killed). No Qt here: this runs in the learner/actor process.
"""

import cProfile
import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

type Job = tuple[Callable[..., object], tuple[object, ...], dict[str, object]]


def main(path: Path, profile_to: Path | None = None) -> None:
    # On Windows the venv's python.exe is a launcher that starts the real interpreter as a child, so the PID the
    # parent sees is the launcher's. Report ours: it is the one py-spy and Task Manager's CPU column care about.
    print(f"interpreter pid {os.getpid()}, python {sys.version.split()[0]}", flush=True)
    with path.open("rb") as f:
        fn, args, kwargs = pickle.load(f)
    path.unlink(missing_ok=True)
    if profile_to is None:
        fn(*args, **kwargs)
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        fn(*args, **kwargs)
    finally:
        profiler.disable()
        profile_to.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_to)
        print(f"profile written to {profile_to}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
