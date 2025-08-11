from __future__ import annotations
from typing import List, Tuple
from traffic_signal import Signal  
from visualize import plot_glosa_diagram
import math, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap, os, json
from utils import *

signals = random_signals(n=5) 

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

# Speed constraints (m/s). Adjust as needed.
v_min = config["speed"]["min_kmh"] / 3.6
v_max = config["speed"]["max_kmh"] / 3.6

# Time horizon (s) for visualization; 3 cycles of the slowest signal is often enough.
t0 = 0.0
horizon_s = max(s.cycle_s for s in signals) * config["visualization"]["time_horizon_cycles"]

# ----------------------
# Run advisory & report
# ----------------------

rec = choose_recommended_speed(signals, t0, v_min, v_max, horizon_s)
v_rec = rec["v"]
kmh = v_rec * 3.6

plot_glosa_diagram(signals, v_rec, kmh, t0, horizon_s)


