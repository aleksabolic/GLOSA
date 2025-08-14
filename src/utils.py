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

def solve_profile_segment(distance, v_initial, time_total, v_max, a_max, n_points=1001,
                          mode="flat", min_exit_speed=None):
    """
    Computes a speed profile for covering a given distance within a specified time.

    Parameters:
        distance (float): Distance to cover.
        v_initial (float): Initial speed.
        time_total (float): Total available time.
        v_max (float): Maximum allowed speed.
        a_max (float): Maximum allowed acceleration.
        n_points (int): Number of discretization points for time.
        mode (str): "time" for time-optimal (maximize exit speed) or "flat" to minimize peak speed.
        min_exit_speed (float, optional): Minimum required exit speed.

    Returns:
        tuple: (time_array, speed_profile, exit_speed) if feasible; otherwise None.
    """
    # Convert parameters to floats for consistency
    a = float(a_max)
    T = float(time_total)
    v0 = float(v_initial)
    V_max = float(v_max)

    # Basic input validation
    if T <= 0 or a <= 0 or V_max < 0 or v0 < 0 or distance < 0:
        return None

    # Discretize the time interval [0, T]
    time_array = np.linspace(0.0, T, n_points)
    # Helper: compute area under speed curve using trapezoidal rule
    area_under = lambda speed_curve: float(np.trapz(speed_curve, time_array))

    def envelope_bounds(v_exit, speed_cap=None):
        """
        Computes the lower and upper bounds (envelopes) for the speed profile.
        
        speed_cap: optional temporary speed limit (must be <= V_max).
        """
        cap = V_max if speed_cap is None else min(V_max, speed_cap)
        upper_bound = np.minimum.reduce([
            np.full(n_points, cap),
            v0 + a * time_array,
            v_exit + a * (T - time_array)
        ])
        lower_bound = np.maximum.reduce([
            np.zeros(n_points),
            v0 - a * time_array,
            v_exit - a * (T - time_array)
        ])
        return lower_bound, upper_bound

    def adjust_speed_from_top(lower_bound, upper_bound, target_area):
        """
        Returns a speed profile by subtracting an optimized constant from the top of the upper_bound,
        ensuring that the area under the profile is as close as possible to target_area.
        """
        lam_low = 0.0
        lam_high = float(np.max(upper_bound - lower_bound))
        for _ in range(60):
            lam = 0.5 * (lam_low + lam_high)
            candidate_speed = np.clip(upper_bound - lam, lower_bound, upper_bound)
            if area_under(candidate_speed) > target_area:
                lam_low = lam
            else:
                lam_high = lam
        return np.clip(upper_bound - lam_high, lower_bound, upper_bound)

    # Compute terminal speed bounds from acceleration constraints
    v_terminal_lower = max(0.0, v0 - a * T)
    v_terminal_upper = min(V_max, v0 + a * T)

    # Quick feasibility check with maximum acceleration and true V_max
    s_max_possible = area_under(
        np.minimum.reduce([
            np.full(n_points, V_max),
            v0 + a * time_array,
            v_terminal_upper + a * (T - time_array)
        ])
    )
    s_min_possible = area_under(
        np.maximum.reduce([
            np.zeros(n_points),
            v0 - a * time_array,
            v_terminal_lower - a * (T - time_array)
        ])
    )
    tolerance = 1e-6 * max(1.0, distance, V_max * T)
    if distance > s_max_possible + tolerance or distance < s_min_possible - tolerance:
        return None

    # Helper: minimal area attainable given a candidate exit speed
    def minimal_area_for_exit(v_exit_candidate):
        lower_env = np.maximum.reduce([
            np.zeros(n_points),
            v0 - a * time_array,
            v_exit_candidate - a * (T - time_array)
        ])
        return area_under(lower_env)

    # Mode "time": aim for maximal exit speed (original behavior)
    if mode == "time":
        if distance >= minimal_area_for_exit(v_terminal_upper) - tolerance:
            v_exit = v_terminal_upper
        else:
            L, R = v_terminal_lower, v_terminal_upper
            for _ in range(60):
                mid = 0.5 * (L + R)
                if minimal_area_for_exit(mid) > distance:
                    R = mid
                else:
                    L = mid
            v_exit = L
        lower_env, upper_env = envelope_bounds(v_exit)
        speed_profile = adjust_speed_from_top(lower_env, upper_env, distance)
        # Enforce boundary conditions
        speed_profile[0] = v0
        speed_profile[-1] = v_exit
        return time_array, speed_profile, v_exit

    # Mode "flat": choose the smallest speed cap covering the distance.
    # Determine required exit speed lower bound
    required_v_exit = v_terminal_lower if min_exit_speed is None else max(v_terminal_lower, min(v_terminal_upper, float(min_exit_speed)))

    def max_area_with_cap(speed_cap):
        """Computes the maximum area achievable under a temporary speed cap."""
        v_exit_cap = min(v_terminal_upper, speed_cap)
        _, up_bound = envelope_bounds(v_exit_cap, speed_cap)
        return area_under(up_bound)

    cap_low = max(0.0, required_v_exit)
    cap_high = V_max
    if distance > max_area_with_cap(cap_high) + tolerance:
        return None  # Infeasible even with full speed cap

    for _ in range(60):
        cap_mid = 0.5 * (cap_low + cap_high)
        if max_area_with_cap(cap_mid) >= distance - tolerance:
            cap_high = cap_mid
        else:
            cap_low = cap_mid
    speed_cap = cap_high

    # Choose an exit speed not exceeding the cap but at least required_v_exit
    v_exit = max(required_v_exit, min(v_terminal_upper, speed_cap))
    lower_env, upper_env = envelope_bounds(v_exit, speed_cap)
    speed_profile = adjust_speed_from_top(lower_env, upper_env, distance)
    speed_profile[0], speed_profile[-1] = v0, v_exit
    return time_array, speed_profile, v_exit

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