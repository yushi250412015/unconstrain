"""Change detectors for the monitoring loop.

Both take a scalar stream and return True on the step they raise an alarm.
They are deliberately conventional -- the question is not whether we can invent
a detector, it is whether the *signal they are given* still contains the change
once sales have been capped by inventory.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PageHinkley:
    """Two-sided Page-Hinkley on the running mean."""

    def __init__(self, delta: float = 0.01, threshold: float = 4.0):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.n = 0
        self.mean = 0.0
        self.cum_up = 0.0
        self.cum_dn = 0.0
        self.max_up = 0.0
        self.min_dn = 0.0

    def reset(self):
        self.__init__(self.delta, self.threshold)

    def update(self, x: float) -> bool:
        x = float(x)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        dev = x - self.mean
        self.cum_up += dev - self.delta
        self.cum_dn += dev + self.delta
        self.max_up = max(self.max_up, self.cum_up)
        self.min_dn = min(self.min_dn, self.cum_dn)
        alarm = ((self.max_up - self.cum_up) > self.threshold
                 or (self.cum_dn - self.min_dn) > self.threshold)
        if alarm:
            self.mean = x
            self.cum_up = self.cum_dn = 0.0
            self.max_up = self.min_dn = 0.0
            self.n = 1
        return alarm


class WindowT:
    """Two-window mean-shift test (a cheap ADWIN).

    Welch-style: compares the newest ``half`` observations against the ``half``
    before them, scaled by the standard error of the difference.  Scale-free on
    purpose -- a bounded-range Hoeffding bound saturates on large shifts, and a
    detector that saturates goes blind exactly when the shift is big.
    """

    def __init__(self, half: int = 14, threshold: float = 3.0):
        self.half = int(half)
        self.threshold = float(threshold)
        self.buf: deque[float] = deque(maxlen=2 * int(half))

    def update(self, x: float) -> bool:
        self.buf.append(float(x))
        if len(self.buf) < 2 * self.half:
            return False
        arr = np.fromiter(self.buf, float, len(self.buf))
        a, b = arr[: self.half], arr[self.half:]
        se = np.sqrt(a.var(ddof=1) / self.half + b.var(ddof=1) / self.half)
        if not np.isfinite(se) or se < 1e-9:
            return False
        return bool(abs(a.mean() - b.mean()) / se > self.threshold)


DETECTORS = {
    "page-hinkley": lambda: PageHinkley(delta=0.005, threshold=7.0),
    "window-t": lambda: WindowT(half=14, threshold=3.5),
}
