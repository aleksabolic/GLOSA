import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Signal:
    position_m: float          # distance from origin (m) along the route
    cycle_s: float             # total cycle length (s)
    green_s: float             # green duration (s)
    yellow_s: float = 3.0      # yellow duration (s) — treated as red for GLOSA
    offset_s: float = 0.0      # time offset (s): at t = 0, green starts at offset_s + k*cycle
    name: str = ""             # optional label

    @property
    def red_s(self) -> float:
        # whatever is left in the cycle
        r = self.cycle_s - self.green_s - self.yellow_s
        return max(0.0, r)

    def phase_at(self, t: float) -> str:
        """Return 'G' during green, else 'R' (yellow treated as red)."""
        tau = (t - self.offset_s) % self.cycle_s
        return 'G' if (0.0 <= tau < self.green_s) else 'R'

    def next_green_start_after(self, t: float) -> float:
        """Absolute time of the next green start after time t (>= t)."""
        tau = (t - self.offset_s) % self.cycle_s
        if tau < self.green_s:
            # already green, next start is now
            return t
        # next cycle's green start
        k = math.floor((t - self.offset_s) / self.cycle_s) + 1
        return self.offset_s + k * self.cycle_s

    def green_windows_between(self, t0: float, t1: float) -> List[Tuple[float, float]]:
        """List of [start, end] absolute green windows intersecting [t0, t1]."""
        if t1 <= t0:
            return []
        # first k such that start_k >= t0 - a tiny epsilon
        k = math.ceil((t0 - self.offset_s - 1e-9) / self.cycle_s)
        windows = []
        while True:
            start = self.offset_s + k * self.cycle_s
            if start > t1:
                break
            end = start + self.green_s
            if end >= t0:
                windows.append((max(start, t0), min(end, t1)))
            k += 1
        return windows
