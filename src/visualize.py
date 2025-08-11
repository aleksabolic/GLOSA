import numpy as np
import matplotlib.pyplot as plt

def plot_glosa_diagram(signals, v_rec, kmh, t0, horizon_s):
    """
    Plots the GLOSA time–distance diagram with green windows for each signal.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    D = max(s.position_m for s in signals)
    xx = np.linspace(0.0, D, 200)
    yy = xx / max(v_rec, 1e-6)  # seconds after t0
    ax.plot(xx, yy, label=f"Recommended ~ {kmh:.1f} km/h")

    for sig in signals:
        greens = sig.green_windows_between(t0, t0 + horizon_s)
        x0 = sig.position_m
        w = max(6.0, 0.01 * D)  # 6 m or 1% of route length
        for (s, e) in greens:
            y0, y1 = (s - t0), (e - t0)
            rect = plt.Rectangle((x0 - 0.5*w, y0), w, (y1 - y0), alpha=0.25, hatch='////')
            ax.add_patch(rect)
        ax.axvline(x=x0, linestyle='--', linewidth=1.0)
        if getattr(sig, 'name', None):
            ax.text(x0, -0.03*horizon_s, sig.name, rotation=90, va='top', ha='center')

    ax.set_xlabel("Distance along route (m)")
    ax.set_ylabel("Time from now (s)")
    ax.set_title("GLOSA Time–Distance Diagram (hatched bars = GREEN windows)")
    ax.set_ylim(0.0, horizon_s)
    ax.set_xlim(0.0, D)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
