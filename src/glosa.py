# Self-contained rerun with speed in km/h.
# Earliest-arrival planning + two animations:
# 1) Road view (car + G/R states)
# 2) Speed vs time in km/h with a moving cursor

import math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Problem setup
distances = [230.0, 280.0, 260.0, 320.0]
cycles   = [60.0, 56.0, 62.0, 58.0]
offsets  = [ 8.0, 12.0,  0.0, 18.0]
g_durs   = [20.0, 18.0, 22.0, 16.0]
windows  = [[(off, off+dur)] for off, dur in zip(offsets, g_durs)]
v_max, a_max = 15.0, 2.0
t0, v0 = 0.0, 5.0

def T_min_link(d, v_in, v_max, a_max):
    t_acc = max(0.0, (v_max - v_in)/a_max)
    s_acc = v_in*t_acc + 0.5*a_max*t_acc**2
    if d <= s_acc:
        return (-v_in + math.sqrt(max(0.0, v_in**2 + 2*a_max*d)))/a_max
    return t_acc + (d - s_acc)/v_max

def earliest_green_at_or_after(t, win_list, C, k_ahead=6):
    base = int(math.floor(t / C))
    best = None
    for k in range(base, base + k_ahead):
        shift = k*C
        for on, off in win_list:
            start, end = on + shift, off + shift
            if t <= end:
                cand = max(t, start)
                if cand <= end and (best is None or cand < best):
                    best = cand
    if best is None:
        raise RuntimeError("no green")
    return best

def solve_profile_segment(d, v_in, dt, v_max, a_max):
    a = 2.0*(d - v_in*dt)/(dt*dt)
    v_out = v_in + a*dt
    if abs(a) <= a_max + 1e-9 and 0.0 <= min(v_in, v_out) and max(v_in, v_out) <= v_max + 1e-9:
        return np.array([0.0, dt]), np.array([v_in, v_out]), v_out
    def s_given_vp(vp):
        if vp >= v_in:
            t1 = (vp - v_in)/a_max
            if t1 > dt: return -1e9
            return (v_in + vp)*0.5*t1 + vp*(dt - t1)
        else:
            t1 = (v_in - vp)/a_max
            if t1 > dt: return -1e9
            return (v_in + vp)*0.5*t1 + vp*(dt - t1)
    lo, hi = 0.0, v_max
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if s_given_vp(mid) < d: lo = mid
        else: hi = mid
    vp = 0.5*(lo+hi)
    if vp >= v_in:
        t1 = (vp - v_in)/a_max
        return np.array([0.0, t1, dt]), np.array([v_in, vp, vp]), vp
    else:
        t1 = (v_in - vp)/a_max
        return np.array([0.0, t1, dt]), np.array([v_in, vp, vp]), vp

# Plan earliest-arrival
cum_dist = [0.0]
for d in distances: cum_dist.append(cum_dist[-1] + d)
t, v = t0, v0
segments, cross_times, cross_speeds = [], [t0], [v0]
for i, d in enumerate(distances):
    t_uncon = t + T_min_link(d, v, v_max, a_max)
    t_cross = earliest_green_at_or_after(t_uncon, windows[i], cycles[i])
    dt = t_cross - t
    seg_t_rel, seg_v, v_exit = solve_profile_segment(d, v, dt, v_max, a_max)
    segments.append((t + seg_t_rel, seg_v))
    t, v = t_cross, v_exit
    cross_times.append(t); cross_speeds.append(v)
total_time = cross_times[-1] - t0

# Build global knots
times = [segments[0][0][0]]; speeds = [segments[0][1][0]]
for seg_t, seg_v in segments:
    if times[-1] == seg_t[0]:
        times.pop(); speeds.pop()
    times.extend(seg_t.tolist()); speeds.extend(seg_v.tolist())
times = np.array(times); speeds = np.array(speeds)

# Helpers
pos_knots = np.zeros_like(times)
for k in range(1, len(times)):
    dt = times[k] - times[k-1]
    pos_knots[k] = pos_knots[k-1] + 0.5*(speeds[k] + speeds[k-1])*dt

def speed_of_t(tq):
    k = np.searchsorted(times, tq, side='right') - 1
    k = max(0, min(k, len(times)-2))
    t0k, t1k = times[k], times[k+1]
    v0k, v1k = speeds[k], speeds[k+1]
    if t1k == t0k: return v1k
    a = (v1k - v0k)/(t1k - t0k)
    return v0k + a*(tq - t0k)

def pos_of_t(tq):
    k = np.searchsorted(times, tq, side='right') - 1
    k = max(0, min(k, len(times)-2))
    t0k, t1k = times[k], times[k+1]
    v0k, v1k = speeds[k], speeds[k+1]
    dt = tq - t0k; segT = t1k - t0k
    a = (v1k - v0k)/segT if segT > 0 else 0.0
    return pos_knots[k] + v0k*dt + 0.5*a*dt*dt

def is_green(i, t_abs):
    C = cycles[i]
    phase = (t_abs % C)
    return any(on <= phase <= off for on, off in windows[i])

# Animations
fps = 60
frames = min(400, max(2, int(np.ceil(total_time*fps))))
t_samples = np.linspace(0.0, total_time, frames)

# 1) Road view
fig1, ax1 = plt.subplots(figsize=(8, 2.8))
ax1.set_xlim(0, cum_dist[-1]); ax1.set_ylim(-1.0, 1.5)
ax1.set_xlabel("position (m)"); ax1.set_yticks([])
ax1.set_title("Car along the corridor with traffic lights")
ax1.plot([0, cum_dist[-1]], [0, 0])
labels = []
for i, x in enumerate(cum_dist[1:], start=1):
    ax1.axvline(x, linestyle="--", linewidth=0.8)
    labels.append(ax1.text(x, 0.8, "?", ha="center", va="center"))
(car_point,) = ax1.plot([0], [0], marker="o", markersize=10)
time_text = ax1.text(0.02*cum_dist[-1], 1.2, "", ha="left", va="center")

def init1():
    car_point.set_data([0], [0]); time_text.set_text("t = 0.0 s")
    for txt in labels: txt.set_text("?")
    return (car_point, time_text, *labels)

def update1(f):
    tq = t_samples[f]
    car_point.set_data([pos_of_t(tq)], [0])
    time_text.set_text(f"t = {tq:5.1f} s")
    for i, _ in enumerate(cum_dist[1:]):
        labels[i].set_text("G" if is_green(i, tq) else "R")
    return (car_point, time_text, *labels)

anim1 = FuncAnimation(fig1, update1, frames=frames, init_func=init1, blit=True, interval=int(1000/fps))

# 2) Speed in km/h
speeds_kmh = 3.6 * speeds
fig2, ax2 = plt.subplots(figsize=(7, 3.2))
ax2.set_xlim(0, total_time); ax2.set_ylim(0, max(1.0, speeds_kmh.max()*1.05))
ax2.set_xlabel("time (s)"); ax2.set_ylabel("speed (km/h)")
ax2.set_title("Speed vs time (km/h)")
ax2.plot(times, speeds_kmh)
(cursor_point,) = ax2.plot([0], [speeds_kmh[0]], marker="o")
cursor_line = ax2.axvline(0.0, linestyle="--", linewidth=0.8)

def init2():
    cursor_point.set_data([0], [speeds_kmh[0]]); cursor_line.set_xdata([0.0, 0.0])
    return (cursor_point, cursor_line)

def update2(f):
    tq = t_samples[f]; vq_kmh = 3.6 * speed_of_t(tq)
    cursor_point.set_data([tq], [vq_kmh]); cursor_line.set_xdata([tq, tq])
    return (cursor_point, cursor_line)

anim2 = FuncAnimation(fig2, update2, frames=frames, init_func=init2, blit=True, interval=int(1000/fps))

plt.show()
