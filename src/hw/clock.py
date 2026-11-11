"""Fixed-rate clock: defines what one env step means in a tunnel that never stops.

Every transition must cover the same dt, otherwise the same deflection held for 3 ms or 7 ms looks identical to the
critic but has different effects. `wait()` blocks until the next tick. A missed tick is counted, never caught up:
firing two steps back to back would be worse than one long step.
"""

import ctypes
import sys
import time

_SPIN_S = 0.0015  # sleep until this close to the deadline, then busy-wait: OS sleep is too coarse to hit it exactly
LATE_TOLERANCE = 0.1  # a tick served up to this fraction of dt late still counts as on time (jitter, not a miss)


class FixedRateClock:
    def __init__(self, dt: float) -> None:
        self.dt = dt
        self._next = 0.0
        self.misses = 0  # ticks skipped because a step overran
        self.last_lateness = 0.0  # how late the most recent tick was served [s]
        self._fine_timer = False

    def start(self) -> None:
        """Start ticking from now. Call at the start of every episode."""
        if sys.platform == "win32" and not self._fine_timer:
            ctypes.windll.winmm.timeBeginPeriod(1)  # Windows sleep granularity 15.6 ms -> 1 ms
            self._fine_timer = True
        self._next = time.perf_counter() + self.dt
        self.misses = 0

    def wait(self) -> None:
        """Block until the next tick."""
        now = time.perf_counter()
        late = now - self._next
        if late > LATE_TOLERANCE * self.dt:
            # Overran: wait for the next tick still in the future. This step becomes one long step and the one after
            # it a full dt again, instead of a long step followed by a short one that catches up.
            skipped = int(late // self.dt) + 1
            self.misses += skipped
            self._next += skipped * self.dt
        if self._next - now > _SPIN_S:
            time.sleep(self._next - now - _SPIN_S)
        while time.perf_counter() < self._next:
            pass
        self.last_lateness = time.perf_counter() - self._next
        self._next += self.dt

    def close(self) -> None:
        if self._fine_timer:
            ctypes.windll.winmm.timeEndPeriod(1)
            self._fine_timer = False
