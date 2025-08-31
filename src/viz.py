import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import blended_transform_factory
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
    # colored markers to indicate current light state (red/green)
    light_markers = []
    for i, x in enumerate(cum_dist[1:], start=1):
        # place a square marker slightly above the road, use PathCollection for easy facecolor updates
        m = ax.scatter([x], [0.6], s=220, c=['gray'], edgecolors='k', zorder=3)
        light_markers.append(m)
        
    car_point, = ax.plot([0], [0], marker="o", markersize=10, label='Optimal')
    # optional greedy car
    greedy_point = None
    if 'greedy' in sim_data:
        greedy_point, = ax.plot([0], [0], marker="s", color='orange', markersize=8, label='Greedy')
    time_text = ax.text(0.02 * cum_dist[-1], 1.2, "", ha="left", va="center")
    
    def init():
        car_point.set_data([0], [0])
        if greedy_point is not None:
            greedy_point.set_data([0], [0])
        time_text.set_text("t = 0.0 s")
        for txt in labels:
            txt.set_text("?")
        # initialize light marker colors at t=0
        for idx, m in enumerate(light_markers):
            col = 'green' if is_green(idx, 0.0, cycles, windows) else 'red'
            m.set_facecolor(col)
        # show legend for optimal vs greedy (if present)
        ax.legend(loc='upper right')
        # include light markers in returned artists so blitting updates their colors
        if greedy_point is not None:
            return (car_point, greedy_point, time_text, *labels, *light_markers)
        return (car_point, time_text, *labels, *light_markers)
    
    def update(f):
        tq = t_samples[f]
        car_point.set_data([pos_func(tq)], [0])
        if greedy_point is not None:
            greedy_pos = sim_data['greedy']['pos_func'](tq)
            greedy_point.set_data([greedy_pos], [0.2])
        time_text.set_text(f"t = {tq:5.1f} s")
        for idx, _ in enumerate(cum_dist[1:]):
            txt = "G" if is_green(idx, tq, cycles, windows) else "R"
            labels[idx].set_text(txt)
            # update colored marker as well
            col = 'green' if is_green(idx, tq, cycles, windows) else 'red'
            light_markers[idx].set_facecolor(col)
        if greedy_point is not None:
            return (car_point, greedy_point, time_text, *labels, *light_markers)
        return (car_point, time_text, *labels, *light_markers)
    
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
    ax.plot(times, speeds, label='Optimal')
    # plot average segment speeds if available
    if avg_speeds is not None:
        ax.plot(times, avg_speeds, c='r', label='Avg segment speed')
    
    cursor_point, = ax.plot([0], [speeds[0]], marker="o")
    greedy_cursor = None
    if 'greedy' in sim_data:
        g = sim_data['greedy']
        ax.plot(g['times'], g['speeds'], c='orange', linestyle='--', label='Greedy')
        greedy_cursor, = ax.plot([0], [g['speeds'][0]], marker='s', color='orange')
    cursor_line = ax.axvline(0.0, linestyle="--", linewidth=0.8)
    
    def init():
        cursor_point.set_data([0], [speeds[0]])
        if greedy_cursor is not None:
            greedy_cursor.set_data([0], [sim_data['greedy']['speeds'][0]])
        cursor_line.set_xdata([0.0, 0.0])
        # legend for speed plot
        ax.legend(loc='upper right')
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

def animate_stacked(sim_data):
    # Unpack
    t = np.asarray(sim_data['t_samples'])
    posf = sim_data['pos_func']
    speedf = sim_data['speed_func']
    times = np.asarray(sim_data['times'])
    speeds = np.asarray(sim_data['speeds'])
    avg_speeds = sim_data.get('avg_speeds')
    cum_dist = np.asarray(sim_data['cum_dist'])
    cycles = sim_data['cycles']
    windows = sim_data['windows']

    # Figure & axes (share time on x between top (primary) & bottom)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[1, 1], constrained_layout=True
    )

    # ---- Top: road view (time axis as primary x), plus twin top x for position ----
    ax_top.set_xlim(t[0], t[-1])
    ax_top.set_ylim(-0.6, 1.6)

    ax_top.set_ylabel("road")
    ax_top.set_yticks([])
    ax_top.set_title("Road view")

    # Twin x-axis for position (plots go on ax_top2)
    ax_top2 = ax_top.twiny()
    ax_top2.set_xlim(cum_dist[0], cum_dist[-1])
    ax_top2.margins(x=0.02)  # add ~2% padding on both sides
    ax_top2.set_xlabel("position (m)")

    # Baseline road on position axis
    ax_top2.plot([cum_dist[0], cum_dist[-1]], [0, 0], lw=1, clip_on=False)

    # Light guides, labels, markers (position-native on ax_top2)
    light_xs = list(cum_dist[1:])
    n_lights = len(light_xs)
    bt = blended_transform_factory(ax_top2.transData, ax_top2.transAxes)
    light_guides = [ax_top2.plot([x, x], [0, 1], ls='--', lw=0.8,
                                transform=bt, clip_on=False, zorder=0, c='C0')[0]
                    for x in light_xs]
    labels = [ax_top2.text(x, 0.8, "?", ha="center", va="center",
                       clip_on=False, zorder=5) for x in light_xs]
    light_markers = [ax_top2.scatter([x], [0.6], s=220, c=['gray'], edgecolors='k',
                                    zorder=6, clip_on=False) for x in light_xs]

    # Car markers (position-native on ax_top2)
    car_opt, = ax_top2.plot([posf(t[0])], [0], marker="o", ms=8, label='Optimal')
    car_greedy = None
    g_posf = None
    g_speedf = None
    if 'greedy' in sim_data:
        g = sim_data['greedy']
        g_posf = g.get('pos_func')
        if g_posf is None:
            if 'positions' in g:
                g_posf = lambda tt, gt=np.asarray(g['times']), gx=np.asarray(g['positions']): np.interp(tt, gt, gx)
            elif 'times' in g and 'speeds' in g:
                gt = np.asarray(g['times'])
                gv = np.asarray(g['speeds']) / 3.6  # km/h -> m/s
                gpos = np.cumsum(np.r_[0.0, 0.5*(gv[1:]+gv[:-1])*(gt[1:]-gt[:-1])])
                g_posf = lambda tt, gt=gt, gpos=gpos: np.interp(tt, gt, gpos)
        g_speedf = g.get('speed_func')
        car_greedy, = ax_top2.plot([g_posf(t[0])], [0.2], marker="s", ms=6, color='orange', label='Greedy')

    # Time label anchored to axes for stability
    time_text = ax_top.text(0.02, 0.9, "", ha="left", va="center", transform=ax_top.transAxes)
    # One legend (put it on ax_top so it doesn't overlap ax_top2 spines)
    handles = [car_opt]
    labels_ = ['Optimal']
    if 'greedy' in sim_data and car_greedy is not None:
        handles.append(car_greedy)
        labels_.append('Greedy')

    leg = ax_top2.legend(handles, labels_, loc='upper right',
                        frameon=True, facecolor='white', edgecolor='0.8')
    leg.set_zorder(10)

    # ---- Bottom: speed vs time (shared x with ax_top primary) ----
    ax_bot.set_xlim(t[0], t[-1])
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("speed (km/h)")
    ax_bot.set_title("Speed vs time (km/h)")

    ax_bot.plot(times, speeds, label='Optimal')
    if avg_speeds is not None:
        ax_bot.plot(times, avg_speeds, c='r', label='Avg segment speed')

    cursor, = ax_bot.plot([t[0]], [speedf(t[0])], marker="o")
    g_cursor = None
    if 'greedy' in sim_data:
        g = sim_data['greedy']
        if 'times' in g and 'speeds' in g:
            ax_bot.plot(g['times'], g['speeds'], c='orange', linestyle='--', label='Greedy')
        g_cursor, = ax_bot.plot([t[0]], [(g_speedf(t[0]) if g_speedf else np.interp(t[0], g['times'], g['speeds']))],
                                marker='s', color='orange')

    vline = ax_bot.axvline(t[0], ls="--", lw=0.8)
    ax_bot.legend(loc='upper right')

    # ---- Animation funcs ----
    def init():
        car_opt.set_data([posf(t[0])], [0])
        if car_greedy is not None:
            car_greedy.set_data([g_posf(t[0])], [0.2])
        time_text.set_text(f"t = {t[0]:.1f} s")
        for i in range(n_lights):
            col = 'green' if is_green(i, 0.0, cycles, windows) else 'red'
            light_markers[i].set_facecolor(col)
            labels[i].set_text('G' if col == 'green' else 'R')
        cursor.set_data([t[0]], [speedf(t[0])])
        if g_cursor is not None:
            g0 = g_speedf(t[0]) if g_speedf else np.interp(t[0], sim_data['greedy']['times'], sim_data['greedy']['speeds'])
            g_cursor.set_data([t[0]], [g0])
        vline.set_xdata([t[0], t[0]])

        artists = [car_opt, time_text, cursor, vline, *labels, *light_markers]
        if car_greedy is not None:
            artists.insert(1, car_greedy)
            if g_cursor is not None:
                artists.insert(3, g_cursor)
        return tuple(artists)

    def update(f):
        tt = t[f]
        # Move cars by their positions (on ax_top2)
        car_opt.set_data([posf(tt)], [0])
        if car_greedy is not None:
            car_greedy.set_data([g_posf(tt)], [0.2])
        time_text.set_text(f"t = {tt:5.1f} s")
        # Update lights’ state at current time
        for i in range(n_lights):
            col = 'green' if is_green(i, tt, cycles, windows) else 'red'
            light_markers[i].set_facecolor(col)
            labels[i].set_text('G' if col == 'green' else 'R')
        # Speed cursors and time cursor
        cursor.set_data([tt], [speedf(tt)])
        if g_cursor is not None:
            gv = g_speedf(tt) if g_speedf else np.interp(tt, sim_data['greedy']['times'], sim_data['greedy']['speeds'])
            g_cursor.set_data([tt], [gv])
        vline.set_xdata([tt, tt])

        artists = [car_opt, time_text, cursor, vline, *labels, *light_markers]
        if car_greedy is not None:
            artists.insert(1, car_greedy)
            if g_cursor is not None:
                artists.insert(3, g_cursor)
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000/60)
    return anim


def show_plots():
    plt.show()