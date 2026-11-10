"""Fixed-rate clock: defines what one env step means in a tunnel that never stops.

Every transition must cover the same dt, otherwise the same deflection held for 3 ms or 7 ms looks identical to the
critic but has different effects. `wait()` blocks until the next tick. A missed tick is counted, never caught up:
firing two steps back to back would be worse than one long step.
"""

import ctypes
import sys
import time

_SPIN_S = 0.0015  # sleep until this close to the deadline, then busy-wait: OS sleep is too coarse to hit it exactly


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
        if now > self._next:
            # Overran: skip to the next tick still in the future instead of rushing to catch up.
            behind = int((now - self._next) // self.dt)
            self.misses += behind
            self._next += behind * self.dt
            if behind:
                now = time.perf_counter()
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
