import math, numpy as np
from utils import (
    T_min_link,
    earliest_green_at_or_after,
    solve_profile_segment,
    build_global_knots,
    build_pos_knots,
    speed_of_t,
    pos_of_t
)
from viz import animate_road_view, animate_speed, show_plots, animate_stacked
from utils import simulate_greedy

# Problem setup
distances = [300.0, 450.0, 280.0, 500.0, 350.0, 420.0, 380.0, 600.0]
cycles    = [65.0, 72.0, 58.0, 80.0, 70.0, 60.0, 75.0, 90.0]
offsets   = [10.0, 20.0, 5.0,  30.0, 12.0, 25.0, 15.0,  40.0]
g_durs    = [25.0, 22.0, 18.0, 28.0, 20.0, 18.0, 24.0,  30.0]


windows  = [[(off, off+dur)] for off, dur in zip(offsets, g_durs)]
v_max, a_max = 60/3.6, 2.0
t0, v0 = 0.0, 0.0

# Plan earliest-arrival
cum_dist = [0.0]
for d in distances:
    cum_dist.append(cum_dist[-1] + d)
t, v = t0, v0
segments = []
avg_segments = []
cross_times = [t0]
cross_speeds = [v0]
for i, d in enumerate(distances):
    t_uncon = t + T_min_link(d, v, v_max, a_max)
    t_cross = earliest_green_at_or_after(t_uncon, windows[i], cycles[i])
    dt = t_cross - t
    res = solve_profile_segment(d, v, dt, v_max, a_max, mode='flat')
    if res is None:
        raise RuntimeError(f"Infeasible segment {i}: d={d}, dt={dt}, v_in={v}, v_max={v_max}, a_max={a_max}")
    seg_t_rel, seg_v, v_exit = res
    segments.append((t + seg_t_rel, seg_v))
    avg_segments.append((t+ seg_t_rel, np.array([d/dt]*len(seg_t_rel))))
    t, v = t_cross, v_exit
    cross_times.append(t)
    cross_speeds.append(v)
total_time = cross_times[-1] - t0

# Build global knots (times and speeds in m/s)
times, speeds = build_global_knots(segments)
# Convert speeds to km/h for display
speeds_kmh = speeds * 3.6

# Build global knots for avg speeds on each segment
_, avg_speeds = build_global_knots(avg_segments)
avg_speeds = avg_speeds * 3.6

# Build position knots (using speeds in m/s)
pos_knots = build_pos_knots(times, speeds)

# Define simulation helper functions that use our global arrays
def sim_speed_of_t(tq):
    # Returns speed in km/h
    return 3.6 * speed_of_t(tq, times, speeds)

def sim_pos_of_t(tq):
    return pos_of_t(tq, times, speeds, pos_knots)

# Time samples for animations
fps = 10
frames = min(400, max(2, int(np.ceil(total_time * fps))))
t_samples = np.linspace(0.0, total_time, frames)

# Prepare simulation data dictionary for visualization
sim_data = {
    'cum_dist': np.array(cum_dist),
    'times': times,
    'speeds': speeds_kmh,   # speeds in km/h for plotting
    'avg_speeds': None, # avg speeds in km/h for plotting
    'pos_func': sim_pos_of_t,
    'speed_func': sim_speed_of_t,
    't_samples': t_samples,
    'total_time': total_time,
    'cycles': cycles,
    'windows': windows
}

# Run greedy driver simulation (discrete-time)
g_times, g_speeds, g_cross_times, g_cross_speeds = simulate_greedy(distances, cycles, windows, v_max, a_max, t0=t0, v0=v0, dt=0.2)

# convert greedy speeds to km/h for plotting
g_speeds_kmh = g_speeds * 3.6

# build greedy position knots for plotting convenience
from utils import build_global_knots, build_pos_knots, speed_of_t, pos_of_t
g_times_knots, g_speeds_knots = build_global_knots([(g_times, g_speeds)])
g_pos_knots = build_pos_knots(g_times_knots, g_speeds_knots)

def sim_g_speed_of_t(tq):
    return 3.6 * speed_of_t(tq, g_times_knots, g_speeds_knots)

def sim_g_pos_of_t(tq):
    return pos_of_t(tq, g_times_knots, g_speeds_knots, g_pos_knots)

# Greedy sim data
sim_data['greedy'] = {
    'times': g_times_knots,
    'speeds': g_speeds_kmh,
    'pos_func': sim_g_pos_of_t,
    'speed_func': sim_g_speed_of_t,
}

# Create animations
# anim1 = animate_road_view(sim_data)
# anim2 = animate_speed(sim_data)
anim3 = animate_stacked(sim_data)
print(f'total time: {total_time}')

# Save plots
# anim1._fig.tight_layout()
# anim1.save("./plots/car_anim.gif", writer='pillow', fps=30)

# anim2._fig.tight_layout()
# anim2.save("./plots/speed_anim.gif", writer='pillow', fps=30)

# Show plots
show_plots()