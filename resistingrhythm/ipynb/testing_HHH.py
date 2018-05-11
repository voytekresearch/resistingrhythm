#%% -------------------------------------------------------------------
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import Range1d
output_notebook()

import numpy as np

from resistingrhythm.util import poisson_impulse
from resistingrhythm.util import poisson_oscillation
from resistingrhythm.util import current_pulse
from resistingrhythm.util import load_spikes

from resistingrhythm.neurons import HHH

#%% -------------------------------------------------------------------
# Shared Params
time = 4
N = 1

V_e = 0
V_i = -80e-3

tau_e = 5e-3
tau_i = 10e-3

tau_h = 2
Ca_target = 0.024

w = (5e-6, 25e-6)

noise_rate = 6  # 20

#%% -------------------------------------------------------------------
# Load data
osc_name = "/Users/type/Code/resistingrhythm/data/osc1180.csv"
stim_name = "/Users/type/Code/resistingrhythm/data/stim0.csv"

ns_osc, ts_osc = load_spikes(osc_name)
ns_in, ts_in = load_spikes(stim_name)

# Plot inputs/osc
p = figure(plot_width=400, plot_height=200)
p.circle(ts_osc, ns_osc, color="grey")
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'N'
p.x_range = Range1d(18, 20)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
show(p)

#%%
# External currents?
# external_current = current_pulse(time, t_onset=2, t_offset=2.5, I=0.5e-6)

#%% -------------------------------------------------------------------
# Run HHH
results = HHH(
    time,
    ns_in,
    ts_in,
    ns_osc,
    ts_osc,
    # np.asarray([]),  # stim
    # np.asarray([]),
    # np.asarray([]),  # osc
    # np.asarray([]),
    external_current=None,
    N=N,
    Ca_target=Ca_target,
    tau_h=tau_h,
    w_in=w,
    tau_in=tau_e,
    V_in=V_e,
    bias_in=0.0e-9,
    w_osc=w,
    tau_osc=tau_i,
    V_osc=V_i,
    noise_rate=noise_rate,
    homeostasis=True,
    time_step=1e-5,
    seed_value=12)

#%%
print(">>> HHH firing rate {}".format(results['ns'].size / N / time))

#%% -------------------------------------------------------------------
# Plot HHH
p = figure(plot_width=600, plot_height=300)
p.circle(results['ts'], results['ns'], color="black")
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'N'
p.x_range = Range1d(0, time)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
show(p)

#%% -------------------------------------------------------------------
p = figure(plot_width=600, plot_height=300)
for n in range(N):
    v = results['v_m'][n, :] * 1e3
    p.line(x=results['times'], y=v, color="black", alpha=0.5)
    print(results['v_m'][n, :].std() * 1e3)
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'V_m (mvolts)'
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.x_range = Range1d(0, time)
show(p)

#%% -------------------------------------------------------------------
p = figure(plot_width=600, plot_height=300)
for n in range(N):
    v = results['I_osc'][n, :]
    p.line(x=results['times'], y=v, color="black", alpha=0.5)
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'I_osc (amps)'
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.x_range = Range1d(0, time)
show(p)

#%% -------------------------------------------------------------------
p = figure(plot_width=600, plot_height=300)
for n in range(N):
    v = results['calcium'][n, :]
    p.line(x=results['times'], y=v, color="black", alpha=0.5)
    p.line(x=results['times'], y=Ca_target, color="purple", alpha=0.2)
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'Ca (moles)'
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.x_range = Range1d(0, 2)
show(p)

#%% -------------------------------------------------------------------
p = figure(plot_width=600, plot_height=300)
for n in range(N):
    p.line(
        x=results['times'],
        y=results['g_KCa'][n, :],
        color="blue",
        alpha=1,
        legend="KCa")
    p.line(
        x=results['times'],
        y=results['g_K'][n, :],
        color="blue",
        alpha=0.5,
        legend="K")
    p.line(
        x=results['times'],
        y=results['g_Na'][n, :],
        color="red",
        alpha=1,
        legend="Na")
    # p.line(
    #     x=results['times'],
    #     y=results['g_Ca'][n, :],
    #     color="purple",
    #     alpha=1,
    #     legend="Ca")
p.xaxis.axis_label = 'Time (s)'
p.yaxis.axis_label = 'g (siemens)'
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.x_range = Range1d(0, 1)
show(p)