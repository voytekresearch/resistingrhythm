from __future__ import division
import csv
import os
import json
import voltagebudget
import numpy as np

from scipy.signal import square
from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def select_n(n, ns, ts):
    m = n == ns
    return ns[m], ts[m]


def mad(x, M=None, axis=None):
    """Mean absolute deviation"""

    if np.isclose(x.size, 0.0):
        return 0.0

    if M is None:
        M = np.mean(x, axis)

    return np.mean(np.absolute(x - M), axis)


def mae(x, y, axis=None):
    """Mean absolute deviation"""

    if np.isclose(x.size, 0.0) and np.isclose(y.size, 0.0):
        return 0.0
    elif np.isclose(x.size, 0.0):
        return np.mean(np.absolute(y))
    elif np.isclose(y.size, 0.0):
        return np.mean(np.absolute(x))

    min_l = min(len(x), len(y))

    x = np.sort(x, axis)
    y = np.sort(y, axis)

    return np.mean(np.absolute(x[:min_l] - y[:min_l]), axis)


def precision(ns, ts, ns_ref, ts_ref, combine=True):
    """Analyze spike time precision (jitter)
    
    Parameters
    ----------
    ns : array-list (1d)
        Neuron codes 
    ts : array-list (1d, seconds)
        Spikes times 
    ns_ref : array-list (1d)
        Neuron codes for the reference train
    ts_ref : array-list (1d, seconds)
        Spikes times for the reference train
    """

    prec = []
    ns_prec = []

    # Join all ns, into the '0' key?
    if combine:
        ns = np.zeros_like(ns)
        ns_ref = np.zeros_like(ns_ref)

    # isolate units, and reformat
    ref = fsutil.to_spikedict(ns_ref, ts_ref)
    target = fsutil.to_spikedict(ns, ts)

    # analyze precision
    for n, r in ref.items():
        try:
            x = target[n]
        except KeyError:
            x = np.zeros_like(r)

        minl = min(len(r), len(x))
        diffs = np.abs([r[i] - x[i] for i in range(minl)])

        prec.append(np.mean(diffs))
        ns_prec.append(n)

    # If were are combining return scalars
    # not sequences
    if combine:
        prec = prec[0]
        ns_prec = ns_prec[0]

    return ns_prec, prec


def index_nearest_spike(ts, t):
    idx = (np.abs(ts - t)).argmin()
    return idx


def nearest_spike(ts, t):
    idx = index_nearest_spike(ts, t)
    return ts[idx]


def create_times(tspan, dt):
    """Define time
    
    Params
    ------
    tspan : tuple (float, float)
        Start and stop times (seconds)
    dt : numeric
        Time step length
    """
    t0 = tspan[0]
    t1 = tspan[1]
    return np.linspace(t0, t1, np.int(np.round((t1 - t0) / dt)))


def step_waves(I, f, duty, t, dt):
    times = fsutil.create_times(t, dt)

    wave = I * square(2 * np.pi * f * times - np.pi, duty=duty)
    wave[wave < 0] = 0.0

    return wave


def find_time_index(times, t):
    idx = (np.abs(times - t)).argmin()
    return idx


def find_time(times, t):
    idx = find_time_index(times, t)
    return times[idx]


def poisson_oscillation(t,
                        t_onset,
                        n_cycles,
                        rate,
                        f,
                        phi=0,
                        n=10,
                        dt=1e-3,
                        min_rate=0.0,
                        seed=None):
    # Define time
    times = fsutil.create_times(t, dt=dt)

    # Create rate wave, over times
    osc = rate / 2 * (1 + np.sin((times * f * 2 * np.pi) + phi))

    # Truncate it to a n_cycle burst, starting at t_onset
    if np.isclose(f, 0.0):
        osc[:] = min_rate
    else:
        burst_l = (1 / float(f)) * n_cycles

        m = np.logical_not(
            np.logical_and(times >= t_onset, times <= (t_onset + burst_l)))
        osc[m] = min_rate

    # Sample osc with N poisson 'cells'
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(osc))

    return ns, ts


def poisson_impulse(t, t_stim, w, rate, n=10, dt=1e-3, seed=None):
    """Create a pulse of spikes w seconds wide, starting at t_stim."""

    # Poisson sample the rate over w
    times = fsutil.create_times(t, dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_stim, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def write_spikes(name, ns, ts):
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(["ns", "ts"])
        writer.writerows([[nrn, spk] for nrn, spk in zip(ns, ts)])


def write_voltages(name, voltages, select=None):
    selected = select_voltages(voltages, select=select)

    for k in selected.keys():

        vs = selected[k]
        # Create header
        head = ",".join([str(n) for n in range(vs.shape[1])])

        # write
        np.savetxt(
            "{}_{}.csv".format(name, k),
            vs.transpose(),
            header=head,
            comments='',
            delimiter=',')
