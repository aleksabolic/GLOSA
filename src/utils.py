import math, numpy as np

def T_min_link(d, v_in, v_max, a_max):
    '''
    minimal time needed to traverse distance d while respecting constraints (v_in, v_max, a_max)
    '''
    t_acc = max(0.0, (v_max - v_in) / a_max)
    s_acc = v_in * t_acc + 0.5 * a_max * t_acc**2
    if d <= s_acc:
        return (-v_in + math.sqrt(max(0.0, v_in**2 + 2 * a_max * d))) / a_max
    return t_acc + (d - s_acc) / v_max

def earliest_green_at_or_after(t, win_list, C, k_ahead=6):
    '''
    earliest green after (>=) time t 
    '''
    base = int(math.floor(t / C))
    best = None
    for k in range(base, base + k_ahead):
        shift = k * C
        for on, off in win_list:
            start, end = on + shift, off + shift
            if t <= end:
                cand = max(t, start)
                if cand <= end and (best is None or cand < best):
                    best = cand
    if best is None:
        raise RuntimeError("no green")
    return best

import numpy as np

def solve_profile_segment(
    d, v_in, dt, v_max, a_max, n=1001,
    mode="flat",          # "flat" (default) or "time"
    v1_min=None           # optional lower bound on exit speed from lookahead
):
    """
    Returns (t, v, v_exit) or None if infeasible.

    mode="flat": minimize peak speed (choose smallest speed cap that still covers d)
    mode="time": maximize exit speed (your current behavior)
    v1_min: required minimum exit speed (from lookahead); if None, no requirement.
    """
    a = float(a_max); T = float(dt); v0 = float(v_in); vmax = float(v_max)
    if T <= 0 or a <= 0 or vmax < 0 or v0 < 0 or d < 0:
        return None

    t = np.linspace(0.0, T, n)
    trapz = lambda x: float(np.trapz(x, t))

    def envelopes(v1, cap=None):
        # cap is an optional temporary speed limit (<= v_max) to “flatten” the profile
        lim = vmax if cap is None else min(vmax, cap)
        up = np.minimum.reduce([np.full(n, lim), v0 + a*t, v1 + a*(T - t)])
        lo = np.maximum.reduce([np.zeros(n),     v0 - a*t, v1 - a*(T - t)])
        return lo, up

    def fill_from_top(lo, up, area):
        lam_lo, lam_hi = 0.0, float(np.max(up - lo))
        for _ in range(60):
            lam = 0.5*(lam_lo + lam_hi)
            v = np.clip(up - lam, lo, up)
            if trapz(v) > area:
                lam_lo = lam
            else:
                lam_hi = lam
        return np.clip(up - lam_hi, lo, up)

    # terminal-speed bounds from accel and v_max
    loB = max(0.0, v0 - a*T)
    hiB = min(vmax, v0 + a*T)

    # quick feasibility with true v_max
    smax = trapz(np.minimum.reduce([np.full(n, vmax), v0 + a*t, hiB + a*(T - t)]))
    smin = trapz(np.maximum.reduce([np.zeros(n),       v0 - a*t, loB - a*(T - t)]))
    epsA = 1e-6 * max(1.0, d, vmax*T)
    if d > smax + epsA or d < smin - epsA:
        return None

    # helper for minimal area attainable for a given terminal speed
    def smin_given(v1):
        lo = np.maximum.reduce([np.zeros(n), v0 - a*t, v1 - a*(T - t)])
        return trapz(lo)

    # --- mode: time-optimal (original behavior) -------------------------------
    if mode == "time":
        if d >= smin_given(hiB) - epsA:
            v1 = hiB
        else:
            L, R = loB, hiB
            for _ in range(60):
                m = 0.5*(L + R)
                if smin_given(m) > d:
                    R = m
                else:
                    L = m
            v1 = L
        lo, up = envelopes(v1)
        v = fill_from_top(lo, up, d)
        v[0], v[-1] = v0, v1
        return t, v, v1

    # --- mode: flat (minimize peak speed subject to covering d) ---------------
    # respect any required exit-speed lower bound
    if v1_min is None:
        v1_req = loB
    else:
        v1_req = max(loB, min(hiB, float(v1_min)))

    # bisection on temporary speed cap c in [c_lo, c_hi] to find the smallest
    # cap that still allows area >= d.
    def smax_with_cap(c):
        # Use the best (largest allowed) v1 under this cap for the "can we reach d?" test
        v1c = min(hiB, c)
        _, upc = envelopes(v1c, cap=c)
        return trapz(upc)

    c_lo = max(0.0, v1_req)   # cap must be at least required exit speed
    c_hi = vmax
    if d > smax_with_cap(c_hi) + epsA:
        return None  # even full v_max cap can't cover distance (shouldn't happen due to earlier check)

    for _ in range(60):
        c_mid = 0.5*(c_lo + c_hi)
        if smax_with_cap(c_mid) >= d - epsA:
            c_hi = c_mid
        else:
            c_lo = c_mid
    cap = c_hi

    # choose an exit speed not exceeding the cap, but >= v1_req
    v1 = max(v1_req, min(hiB, cap))

    lo, up = envelopes(v1, cap=cap)
    v = fill_from_top(lo, up, d)
    v[0], v[-1] = v0, v1
    return t, v, v1


def build_global_knots(segments):
    times = [segments[0][0][0]]
    speeds = [segments[0][1][0]]
    for seg_t, seg_v in segments:
        if times[-1] == seg_t[0]:
            times.pop()
            speeds.pop()
        times.extend(seg_t.tolist())
        speeds.extend(seg_v.tolist())
    return np.array(times), np.array(speeds)

def build_pos_knots(times, speeds):
    pos_knots = np.zeros_like(times)
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        pos_knots[k] = pos_knots[k - 1] + 0.5 * (speeds[k] + speeds[k - 1]) * dt
    return pos_knots

def speed_of_t(tq, times, speeds):
    k = np.searchsorted(times, tq, side='right') - 1
    k = max(0, min(k, len(times) - 2))
    t0k, t1k = times[k], times[k + 1]
    v0k, v1k = speeds[k], speeds[k + 1]
    if t1k == t0k:
        return v1k
    a = (v1k - v0k) / (t1k - t0k)
    return v0k + a * (tq - t0k)

def pos_of_t(tq, times, speeds, pos_knots):
    k = np.searchsorted(times, tq, side='right') - 1
    k = max(0, min(k, len(times) - 2))
    t0k, t1k = times[k], times[k + 1]
    v0k, v1k = speeds[k], speeds[k + 1]
    dt = tq - t0k
    segT = t1k - t0k
    a = (v1k - v0k) / segT if segT > 0 else 0.0
    return pos_knots[k] + v0k * dt + 0.5 * a * dt * dt

def is_green(i, t_abs, cycles, windows):
    C = cycles[i]
    phase = t_abs % C
    return any(on <= phase <= off for on, off in windows[i])