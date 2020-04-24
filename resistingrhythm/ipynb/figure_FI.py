#!/usr/bin/env python
# coding: utf-8

# # FI curves
#
# Show explicit changes in excitbility with H

# In[8]:

# import matplotlib.pyplot as plt
# # plt.rcParams.update({'font.size': 50})

# get_ipython().run_line_magic('matplotlib', 'inline')

# import seaborn as sns
# sns.set(font_scale=3)
# sns.set_style('ticks', {'axes.edgecolor': '0'})

import numpy as np
import pandas as pd

from resistingrhythm.util import poisson_impulse
from resistingrhythm.util import poisson_oscillation
from resistingrhythm.util import current_pulse
from resistingrhythm.util import load_spikes
from resistingrhythm.util import create_times
from resistingrhythm.neurons import HHH

from fakespikes.rates import square_pulse

# # Simulation
#
# Run twice. Once with osc_rate = 0. Once with osc_rate = 1.5

# ## Tune

# # In[ ]:

# # Shared parameters
# time = 4
# N = 100
# tau_h = 1

# V_e = 0
# tau_e = 5e-3
# w = (5e-6, 50e-6)

# dt = 1e-5

# # In[ ]:

# # Load some input
# osc_name = "/Users/qualia/Code/resistingrhythm/data/osc115.csv"
# stim_name = "/Users/qualia/Code/resistingrhythm/data/stim3.csv"

# # ns_osc, ts_osc = np.asarray([]), np.asarray([])
# ns_osc, ts_osc = load_spikes(osc_name)

# # -
# # plt.figure()
# fig, ax = plt.subplots(1, sharex=True, figsize=(10, 2))
# _ = ax.plot(ts_osc, ns_osc, markersize=.5, marker='o', linestyle='', color='k')
# _ = ax.set_xlim(0, 2)
# sns.despine()

# # In[ ]:

# # Create square wave
# a = 0.6e-5
# t_pulse = 3.75
# w_pulse = .1
# # t_pulse = 19.5
# # w_pulse = .1

# times = create_times((0, time), dt=dt)
# Is = square_pulse(times, a, t_pulse, w_pulse, dt=dt, min_a=0)

# # -
# fig, ax = plt.subplots(1, sharex=True, figsize=(2, 3))
# _ = ax.plot(times, Is, color='k')
# # _ = ax.set_xlim(19, 20.0)
# sns.despine()

# # In[ ]:

# # Run 1
# results_1 = HHH(
#     time,
#     np.asarray([]), # No spike input
#     np.asarray([]),
#     ns_osc,
#     ts_osc,
#     external_current=Is,
#     N=N,
#     Ca_target=0.003,
#     tau_h=tau_h,
#     w_in=w,
#     tau_in=tau_e,
#     V_in=V_e,
#     bias_in=0.0e-9,
#     w_osc=w,
#     tau_osc=tau_e,
#     V_osc=V_e,
#     sigma=0,
#     homeostasis=False,
#     time_step=dt,
#     seed_value=42
# )

# # In[ ]:

# # Run 2
# results_2 = HHH(
#     time,
#     np.asarray([]), # No spike input
#     np.asarray([]),
#     ns_osc,
#     ts_osc,
#     external_current=Is,
#     N=N,
#     Ca_target=0.003,
#     tau_h=tau_h,
#     w_in=w,
#     tau_in=tau_e,
#     V_in=V_e,
#     bias_in=0.0e-9,
#     w_osc=w,
#     tau_osc=tau_e,
#     V_osc=V_e,
#     sigma=0,
#     homeostasis=True,
#     time_step=dt,
#     seed_value=42
# )

# # In[ ]:

# # -
# fig, ax = plt.subplots(2, sharex=True, figsize=(10, 3))
# _ = ax[0].plot(results_1['ts'], results_1['ns'], marker='o', linestyle='', color='k')
# _ = ax[1].plot(results_2['ts'], results_2['ns'], marker='o', linestyle='', color='k')
# _ = ax[0].set_xlim(0, 4)
# sns.despine()

# # In[ ]:

# # -
# fig, ax = plt.subplots(2, sharex=True, figsize=(10, 6))
# _ = ax[0].plot(results_1['times'], results_1['v_m'][0,:], linestyle='-', color='k')
# _ = ax[1].plot(results_2['times'], results_2['v_m'][0,:], linestyle='-', color='k')
# _ = ax[0].set_xlim(3.7, 3.9)
# sns.despine()

# # In[ ]:

# # Est rate
# rate_1 = (results_1['ts'] > 3.5).sum()
# rate_2 = (results_2['ts'] > 3.5).sum()
# print(f"Rate 1: {rate_1}, Rate 2: {rate_2}")

# # Generate curves

# In[2]:

# Load some input
# osc_name = "/Users/qualia/Code/resistingrhythm/data/osc115.csv"
osc_name = "/home/ejp/src/resistingrhythm/data/osc115.csv"
ns_osc, ts_osc = load_spikes(osc_name)

# Drop anything after 4 seconds
m = ts_osc > 4
ns_osc, ts_osc = ns_osc[m], ts_osc[m]

# Shared parameters
time = 5
N = 100
tau_h = 1

V_e = 0
tau_e = 5e-3
w = (5e-6, 50e-6)

dt = 1e-5

# Impulse desgin
t_pulse = 4.5
w_pulse = .1
a_start = 0.1e-5
a_stop = 1.0e-5
n_steps = 90
a_range = np.linspace(a_start, a_stop, n_steps)

# -
rates_1, rates_2 = [], []
for i, a in enumerate(a_range):
    print(f"Curve: {i}. Pulse: {a}")

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
    r_1 = (results_1['ts'] > 4.5).mean()
    r_2 = (results_2['ts'] > 4.5).mean()
    s_1 = (results_1['ts'] > 4.5).std()
    s_2 = (results_2['ts'] > 4.5).std()
    rates_1.append(r_1)
    rates_2.append(r_2)
    vars_1.append(s_1)
    vars_2.append(s_2)
    
    # Progress bar....
    print(f"Rate 1 (no): {r_1}. Rate 2 (yes): {r_2}")

# In[16]:

fi = {
    'rate_ref': rates_1, 
    'rate_h': rates_2, 
    'var_ref' : vars_1,
    'var_h' : vars_2,
    'impulse': a_range}

# to npy
np.save('FI_curves', fi)
# to csv
pd.DataFrame(fi).to_csv('FI_curves.csv', index=False)

# In[ ]:

# fig, ax = plt.subplots(1, sharex=True, figsize=(7, 4))
# _ = ax.plot(
#     fi['impulse'] * 1e6,
#     np.asarray(fi['rate_no']) / 100,
#     color='lightgrey',
#     linestyle="--",
#     linewidth=4)
# _ = ax.plot(
#     fi['impulse'] * 1e6,
#     np.asarray(fi['rate_h']) / 100,
#     color='black',
#     linewidth=4)
# _ = ax.set_xlabel("Impulse (usiemens)")
# _ = ax.set_ylabel("Avg. firing (Hz)")
# sns.despine()

# In[ ]:
