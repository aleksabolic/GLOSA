from typing import List, Tuple
import numpy as np
from traffic_signal import Signal
import yaml
import random

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intervals = [(a, b) for (a, b) in intervals if a < b]
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + 1e-9:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def intersect_interval_sets(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Intersection of two union-of-intervals."""
    i, j, out = 0, 0, []
    while i < len(A) and j < len(B):
        a1, a2 = A[i]
        b1, b2 = B[j]
        lo = max(a1, b1)
        hi = min(a2, b2)
        if lo < hi - 1e-12:
            out.append((lo, hi))
        if a2 < b2:
            i += 1
        else:
            j += 1
    return out

def speeds_allowing_green_at_light(sig: Signal, t0: float, vmin: float, vmax: float, horizon: float) -> List[Tuple[float, float]]:
    """For a single light, compute all constant-speed intervals [v1,v2] ⊆ [vmin,vmax] that arrive during green within horizon."""
    d = sig.position_m
    windows = sig.green_windows_between(t0, t0 + horizon)
    out = []
    for (s, e) in windows:
        # Arrival time t_arr = t0 + d / v ∈ [s, e] ⇒ v ∈ [d/(e-t0),  d/max(s-t0, 0+)]
        # If s <= t0, upper bound is +∞ (we can still catch current green if we can reach before e).
        denom_lo = (e - t0)
        if denom_lo <= 0:
            continue  # window already ended
        v_lo = d / denom_lo
        if s <= t0:
            v_hi = float('inf')
        else:
            v_hi = d / (s - t0)
        lo = max(vmin, v_lo)
        hi = min(vmax, v_hi)
        if lo < hi:
            out.append((lo, hi))
    return merge_intervals(out)

def feasible_speeds_all_lights(signals: List[Signal], t0: float, vmin: float, vmax: float, horizon: float) -> List[Tuple[float, float]]:
    # Start with [vmin, vmax], intersect with each signal's feasible set
    feas = [(vmin, vmax)]
    for sig in signals:
        S = speeds_allowing_green_at_light(sig, t0, vmin, vmax, horizon)
        feas = intersect_interval_sets(feas, S)
        if not feas:
            break
    return merge_intervals(feas)

def total_wait_time_at_speed(signals: List[Signal], t0: float, v: float) -> float:
    wait = 0.0
    for sig in signals:
        t_arr = t0 + sig.position_m / v
        if sig.phase_at(t_arr) == 'G':
            continue
        t_next_g = sig.next_green_start_after(t_arr)
        wait += (t_next_g - t_arr)
    return wait

def choose_recommended_speed(signals: List[Signal], t0: float, vmin: float, vmax: float, horizon: float) -> dict:
    feas = feasible_speeds_all_lights(signals, t0, vmin, vmax, horizon)
    if feas:
        # pick center of widest feasible band
        widths = [b - a for a, b in feas]
        i = int(np.argmax(widths))
        a, b = feas[i]
        v_star = 0.5 * (a + b)
        mode = "nonstop"
        return {"v": v_star, "mode": mode, "feasible_bands": feas}
    # else minimize total waiting time over a grid
    grid = np.linspace(vmin, vmax, 200)
    waits = np.array([total_wait_time_at_speed(signals, t0, float(v)) for v in grid])
    j = int(np.argmin(waits))
    v_star = float(grid[j])
    mode = "min_wait"
    return {"v": v_star, "mode": mode, "feasible_bands": [], "grid": grid, "waits": waits}

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def random_signals(
    n: int,
    min_distance: float = 300.0,
    max_distance: float = 2000.0,
    min_cycle: float = 60.0,
    max_cycle: float = 120.0,
    min_green: float = 20.0,
    max_green: float = 60.0,
    min_yellow: float = 3.0,
    max_yellow: float = 6.0,
) -> list[Signal]:
    positions = sorted(random.uniform(min_distance, max_distance) for _ in range(n))
    signals = []
    for i, pos in enumerate(positions):
        cycle = random.uniform(min_cycle, max_cycle)
        green = random.uniform(min_green, max_green)
        yellow = random.uniform(min_yellow, max_yellow)
        offset = random.uniform(0, cycle)
        signals.append(
            Signal(
                position_m=pos,
                cycle_s=cycle,
                green_s=green,
                yellow_s=yellow,
                offset_s=offset,
                name=f"S{i+1}"
            )
        )
    return signals
