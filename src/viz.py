import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from utils import pos_of_t, speed_of_t, is_green

def animate_road_view(sim_data):
    cum_dist = sim_data['cum_dist']
    pos_func = sim_data['pos_func']
    cycles = sim_data['cycles']
    windows = sim_data['windows']
    t_samples = sim_data['t_samples']
    
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0, cum_dist[-1])
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel("position (m)")
    ax.set_yticks([])
    ax.set_title("Car along the corridor with traffic lights")
    ax.plot([0, cum_dist[-1]], [0, 0])
    
    labels = []
    for i, x in enumerate(cum_dist[1:], start=1):
        ax.axvline(x, linestyle="--", linewidth=0.8)
        labels.append(ax.text(x, 0.8, "?", ha="center", va="center"))
        
    car_point, = ax.plot([0], [0], marker="o", markersize=10)
    # optional greedy car
    greedy_point = None
    if 'greedy' in sim_data:
        greedy_point, = ax.plot([0], [0], marker="s", color='orange', markersize=8)
    time_text = ax.text(0.02 * cum_dist[-1], 1.2, "", ha="left", va="center")
    
    def init():
        car_point.set_data([0], [0])
        if greedy_point is not None:
            greedy_point.set_data([0], [0])
        time_text.set_text("t = 0.0 s")
        for txt in labels:
            txt.set_text("?")
        if greedy_point is not None:
            return (car_point, greedy_point, time_text, *labels)
        return (car_point, time_text, *labels)
    
    def update(f):
        tq = t_samples[f]
        car_point.set_data([pos_func(tq)], [0])
        if greedy_point is not None:
            greedy_pos = sim_data['greedy']['pos_func'](tq)
            greedy_point.set_data([greedy_pos], [0.2])
        time_text.set_text(f"t = {tq:5.1f} s")
        for i, _ in enumerate(cum_dist[1:]):
            txt = "G" if is_green(i, tq, cycles, windows) else "R"
            labels[i].set_text(txt)
        if greedy_point is not None:
            return (car_point, greedy_point, time_text, *labels)
        return (car_point, time_text, *labels)
    
    anim = FuncAnimation(fig, update, frames=len(t_samples), init_func=init, blit=True, interval=1000/60)
    return anim

def animate_speed(sim_data):
    times = sim_data['times']
    speeds = sim_data['speeds']  # speeds already in km/h
    avg_speeds = sim_data['avg_speeds']
    t_samples = sim_data['t_samples']
    
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, sim_data['total_time'])
    ax.set_ylim(0, max(1.0, speeds.max() * 1.05))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title("Speed vs time (km/h)")
    ax.plot(times, speeds)
    
    cursor_point, = ax.plot([0], [speeds[0]], marker="o")
    greedy_cursor = None
    if 'greedy' in sim_data:
        g = sim_data['greedy']
        ax.plot(g['times'], g['speeds'], c='orange', linestyle='--')
        greedy_cursor, = ax.plot([0], [g['speeds'][0]], marker='s', color='orange')
    cursor_line = ax.axvline(0.0, linestyle="--", linewidth=0.8)
    
    def init():
        cursor_point.set_data([0], [speeds[0]])
        if greedy_cursor is not None:
            greedy_cursor.set_data([0], [sim_data['greedy']['speeds'][0]])
        cursor_line.set_xdata([0.0, 0.0])
        if greedy_cursor is not None:
            return (cursor_point, greedy_cursor, cursor_line)
        return (cursor_point, cursor_line)
    
    def update(f):
        tq = t_samples[f]
        vq = sim_data['speed_func'](tq)
        cursor_point.set_data([tq], [vq])
        if greedy_cursor is not None:
            gv = sim_data['greedy']['speed_func'](tq)
            greedy_cursor.set_data([tq], [gv])
        cursor_line.set_xdata([tq, tq])
        if greedy_cursor is not None:
            return (cursor_point, greedy_cursor, cursor_line)
        return (cursor_point, cursor_line)
    
    anim = FuncAnimation(fig, update, frames=len(t_samples), init_func=init, blit=True, interval=1000/60)
    return anim

def show_plots():
    plt.show()