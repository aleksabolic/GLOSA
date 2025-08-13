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

def solve_profile_segment(d, v_in, dt, v_max, a_max, n=1001):
    """
    Returns (t, v, v_exit):
      t: time samples in [0, dt]
      v: velocity samples respecting 0<=v<=v_max and |dv/dt|<=a_max
      v_exit: maximal feasible terminal speed
    Returns None if infeasible.
    """
    a = float(a_max); T = float(dt); v0 = float(v_in); vmax = float(v_max)
    if T <= 0 or a <= 0 or vmax < 0 or v0 < 0 or d < 0: return None
    t = np.linspace(0.0, T, n)

    def envelopes(v1):
        up = np.minimum.reduce([np.full(n, vmax), v0 + a*t, v1 + a*(T - t)])
        lo = np.maximum.reduce([np.zeros(n),        v0 - a*t, v1 - a*(T - t)])
        return lo, up

    trapz = lambda x: float(np.trapz(x, t))

    def feasible(v1):
        if v1 < 0 or v1 > vmax or abs(v1 - v0) > a*T: return False
        lo, up = envelopes(v1)
        return trapz(lo) - 1e-9 <= d <= trapz(up) + 1e-9

    # --- choose v1 (maximize terminal speed) ---
    loB = max(0.0, v0 - a*T)            # min reachable end speed
    hiB = min(vmax, v0 + a*T)           # max reachable end speed

    # quick impossibility check: even with best case v1=hiB and upper envelope, can't reach d
    def smax_given(v1):
        up = np.minimum.reduce([np.full(n, vmax), v0 + a*t, v1 + a*(T - t)])
        return trapz(up)

    def smin_given(v1):
        lo = np.maximum.reduce([np.zeros(n), v0 - a*t, v1 - a*(T - t)])
        return trapz(lo)

    if smax_given(hiB) + 1e-9 < d:
        return None  # truly impossible segment

    if feasible(hiB):
        v1 = hiB  # best case: take the top reachable exit speed
    else:
        # Here d < smin(hiB); lower v1 until smin(v1) <= d (monotone in v1)
        L, R = loB, hiB
        for _ in range(60):
            m = 0.5*(L + R)
            if smin_given(m) > d:   # still too much area even at minimal profile → lower v1
                R = m
            else:
                L = m
        v1 = L

    lo, up = envelopes(v1)

    # choose λ so ∫ clip(up-λ, lo, up) dt = d  (water-filling)
    lam_lo, lam_hi = 0.0, float(np.max(up - lo))
    for _ in range(60):
        lam = 0.5*(lam_lo + lam_hi)
        v = np.clip(up - lam, lo, up)
        if np.trapz(v, t) > d: lam_lo = lam
        else: lam_hi = lam
    v = np.clip(up - lam_hi, lo, up)
    v[0], v[-1] = v0, v1  # clean endpoints

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