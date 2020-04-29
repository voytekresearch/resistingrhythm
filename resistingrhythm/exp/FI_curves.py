#!/usr/bin/env python
# coding: utf-8
"""Generate FI curves, with and without H"""

import numpy as np
import pandas as pd

from resistingrhythm.util import poisson_impulse
from resistingrhythm.util import poisson_oscillation
from resistingrhythm.util import current_pulse
from resistingrhythm.util import load_spikes
from resistingrhythm.util import create_times
from resistingrhythm.neurons import HHH

from fakespikes.rates import square_pulse

# Load some input; set path for data by hand....
osc_name = "/Users/qualia/Code/resistingrhythm/data/osc115.csv"
# osc_name = "/home/ejp/src/resistingrhythm/data/osc115.csv"
# osc_name = "/home/stitch/Code/resistingrhythm/data/osc115.csv"

ns_osc, ts_osc = load_spikes(osc_name)

# Drop anything after 4 seconds
m = ts_osc > 4
ns_osc, ts_osc = ns_osc[m], ts_osc[m]

# Shared parameters
time = 5
dt = 1e-5

N = 100
tau_h = 1
V_e = 0
tau_e = 5e-3
w = (5e-6, 50e-6)

# Impulse desgin
t_pulse = 4.5
w_pulse = .1
a_start = 0.01e-5
a_stop = 1.0e-5
n_steps = 2
a_range = np.linspace(a_start, a_stop, n_steps)

# Num replicates
num_iter = 2

# -
iterations = []
rates_1, rates_2 = [], []

for n in range(num_iter):
    for i, a in enumerate(a_range):
        print(f"Iteration {n} - Curve: {i}. Pulse: {a}")

        # Create pulse
        times = create_times((0, time), dt=dt)
        Is = square_pulse(times, a, t_pulse, w_pulse, dt=dt, min_a=0)

        # Run 1
        results_1 = HHH(
            time,
            np.asarray([]),  # No spike input
            np.asarray([]),
            ns_osc,
            ts_osc,
            external_current=Is,
            N=N,
            Ca_target=0.003,
            tau_h=tau_h,
            w_in=w,
            tau_in=tau_e,
            V_in=V_e,
            bias_in=0.0e-9,
            w_osc=w,
            tau_osc=tau_e,
            V_osc=V_e,
            sigma=0,
            homeostasis=False,
            time_step=dt,
            seed_value=42)

        # Run 2
        results_2 = HHH(
            time,
            np.asarray([]),  # No spike input
            np.asarray([]),
            ns_osc,
            ts_osc,
            external_current=Is,
            N=N,
            Ca_target=0.003,
            tau_h=tau_h,
            w_in=w,
            tau_in=tau_e,
            V_in=V_e,
            bias_in=0.0e-9,
            w_osc=w,
            tau_osc=tau_e,
            V_osc=V_e,
            sigma=0,
            homeostasis=True,
            time_step=dt,
            seed_value=42)

        # Extract rates and SD, and save
        r_1 = (results_1['ts'] > 4.5).sum()
        r_2 = (results_2['ts'] > 4.5).sum()

        # Save
        iterations.append(n)
        rates_1.append(r_1)
        rates_2.append(r_2)

        # Progress bar....
        print(f"Rate 1 (no): {r_1}. Rate 2 (yes): {r_2}")

# -
# Save the curves to both an npy and a .csv
fi = {
    'rate_ref': rates_1,
    'rate_h': rates_2,
    'n': iterations,
    'impulse': a_range.tolist() * num_iter
}

# to npy
np.save('FI_curves', fi)
# to csv
pd.DataFrame(fi).to_csv('FI_curves.csv', index=False)