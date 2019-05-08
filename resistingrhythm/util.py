from __future__ import division
import csv
import os
import numpy as np

from scipy.signal import square
from fakespikes import neurons, rates
from fakespikes import util as fsutil

import resistingrhythm

import numpy as np
from scipy import signal


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def select_n(n, ns, ts):
    m = n == ns
    return ns[m], ts[m]


def l1_by_n(N, ns_ref, ts_ref, ns_n, ts_n):
    v_i = []
    v_r = []
    e_i = []
    for i in range(N):
        ns_ref_i, ts_ref_i = select_n(i, ns_ref, ts_ref)
        ns_i, ts_i = select_n(i, ns_n, ts_n)

        # Variance
        if ts_i.size > 0:
            var = mad(ts_i) / ts_i.size
        else:
            var = 0
        v_i.append(var)

        if ts_ref_i.size > 0:
            var = mad(ts_ref_i) / ts_ref_i.size
        else:
            var = 0
        v_r.append(var)

        # Error
        e_i.append(mae(ts_i, ts_ref_i))

    # Expectation of all neurons.
    var = np.mean(v_i)
    var_ref = np.mean(v_r)
    error = np.mean(e_i)

    return var, var_ref, error


def l2_by_n(N, ns_ref, ts_ref, ns_n, ts_n):
    v_i = []
    v_r = []
    e_i = []
    for i in range(N):
        ns_ref_i, ts_ref_i = select_n(i, ns_ref, ts_ref)
        ns_i, ts_i = select_n(i, ns_n, ts_n)

        # Variance
        if ts_i.size > 0:
            var = np.std(ts_i)
        else:
            var = 0
        v_i.append(var)

        if ts_ref_i.size > 0:
            var = np.std(ts_ref_i)
        else:
            var = 0
        v_r.append(var)

        # Error
        e_i.append(mse(ts_i, ts_ref_i))

    # Expectation of all neurons.
    var = np.mean(v_i)
    var_ref = np.mean(v_r)
    error = np.mean(e_i)

    return var, var_ref, error


def mad(x, M=None, axis=None):
    """Mean absolute deviation"""

    if np.isclose(x.size, 0.0):
        return 0.0

    if M is None:
        M = np.mean(x, axis)

    return np.mean(np.absolute(x - M), axis)


def mae(x, y, axis=None):
    """Mean absolute error"""

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


def mse(x, y, axis=None):
    """Mean squared error"""

    if np.isclose(x.size, 0.0) and np.isclose(y.size, 0.0):
        return 0.0
    elif np.isclose(x.size, 0.0):
        return np.mean(np.absolute(y))
    elif np.isclose(y.size, 0.0):
        return np.mean(np.absolute(x))

    min_l = min(len(x), len(y))

    x = np.sort(x, axis)
    y = np.sort(y, axis)

    return np.mean((x[:min_l] - y[:min_l])**2, axis)


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


def _pulse(I, on, off, t, dt):
    times = create_times((0, t), dt)
    wave = np.zeros_like(times)
    i = find_time_index(times, on)
    j = find_time_index(times, off)
    wave[i:j] = I

    return wave


def current_pulse(t=1, t_onset=0.8, t_offset=0.9, I=10e-9, dt=1e-5, name=None):
    wave = _pulse(I, t_onset, t_offset, t, dt)

    if name is not None:
        np.savetxt("{}.csv".format(name), wave.transpose(), delimiter=',')
        return None
    else:
        return wave


def load_current(filename):
    x = np.loadtxt(filename)
    return x


def poisson_hippocampus_theta(t=1,
                              t_onset=0.2,
                              n_cycles=8,
                              rate=6,
                              n=10,
                              dt=1e-3,
                              name=None,
                              seed=None):
    """Simulate a Poisson population based on LFP from CA1."""
    # Requested rate
    sampling_rate = int(1 / dt)

    # Load LFP
    base_path = os.path.split(resistingrhythm.__file__)[0]
    lfp_raw = np.load(os.path.join(base_path, 'data/ca1.npy'))
    native_sampling_rate = int(1252)  # sampling rate in ../data/ca1.npy

    # Limit to t; lfp_raw is 41 Mb on disk.
    times = np.arange(0,
                      len(lfp_raw) / native_sampling_rate,
                      1 / native_sampling_rate)
    lfp_raw = lfp_raw[times <= t]

    # Resample lfp?
    if native_sampling_rate != sampling_rate:
        lfp = signal.resample(
            lfp_raw, int(
                (lfp_raw.size / native_sampling_rate) * sampling_rate))
    else:
        lfp = lfp_raw

    # Normalize lfp -> (0, 1)
    lfp = (lfp - np.min(lfp)) / (np.max(lfp) - np.min(lfp))

    # Recreate time
    times = np.arange(0, len(lfp) / sampling_rate, 1 / sampling_rate)

    # Truncate lFP to burst parameters.
    f = 8  # fixed here; approximate.
    burst_l = (1 / float(f)) * n_cycles
    m = np.logical_not(
        np.logical_and(times >= t_onset, times <= (t_onset + burst_l)))
    lfp[m] = 0

    # Make spikes
    nrns = neurons.Spikes(n, t, dt=dt, refractory=dt, seed=None)
    firing_rate = nrns.poisson(rate * lfp)
    ns, ts = fsutil.to_spiketimes(times[0:-1], firing_rate)

    if name is not None:
        write_spikes(name, ns, ts)
        return None
    else:
        return ns, ts


def poisson_oscillation(t=1,
                        t_onset=0.2,
                        n_cycles=10,
                        rate=6,
                        f=8,
                        phi=0,
                        n=10,
                        dt=1e-3,
                        min_rate=0.0,
                        name=None,
                        seed=None):
    """Simulate a Poisson population, oscillating"""

    # Define time
    times = fsutil.create_times(t, dt=dt)

    # Create rate wave, over times
    osc = (rate / 2 * (1 + np.sin((times * f * 2 * np.pi) + phi))) + min_rate

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

    if name is not None:
        write_spikes(name, ns, ts)
        return None
    else:
        return ns, ts


def poisson_drift_rate(t=1,
                       t_onset=0.2,
                       tau=.5,
                       sigma=0.01,
                       rate=6.0,
                       n=10,
                       dt=1e-3,
                       min_rate=0,
                       name=None,
                       seed=None):
    """Simulate a Poisson population with an OU rate drift"""
    prng = np.random.RandomState(seed)

    times = create_times((0, t), dt)

    # OU as a difference equation
    x = min_rate
    rates = [x]
    for t in times[1:]:
        if t < t_onset:
            rates.append(x)
            continue

        # Last x
        xi = prng.normal(0, 1)

        # Drift, scaling the step by sigma * sqrt(tau)
        delta_x = (sigma * np.sqrt(tau) * xi) / tau
        x -= delta_x

        # Save
        rates.append(x)

    # Drop initial value
    rates = np.asarray(rates)

    # No negative rates
    rates[rates < min_rate] = min_rate

    # Sample xs with N poisson 'cells'
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(rates))

    # -
    if name is not None:
        write_spikes(name, ns, ts)
        return None
    else:
        return ns, ts, times, rates


def poisson_impulse(t=1,
                    t_onset=0.2,
                    w=1,
                    rate=6,
                    n=10,
                    dt=1e-3,
                    name=None,
                    seed=None):
    """Simulate a pulse of spikes w seconds wide, starting at t_onset."""

    # Poisson sample the rate over w
    times = create_times((0, t), dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_onset, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    if name is not None:
        write_spikes(name, ns, ts)
        return None
    else:
        return ns, ts


def write_spikes(name, ns, ts):
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(["ns", "ts"])
        writer.writerows([[nrn, spk] for nrn, spk in zip(ns, ts)])


def _read_csv_cols_into_dict(filename):
    # csv goes here:
    data = {}

    # Open and iterate over lines
    # when csv is returning lines as dicts
    # using the header as a key
    reader = csv.DictReader(open(filename, 'r'))
    for row in reader:
        for k, v in row.items():
            # Is a number?
            try:
                v = float(v)
            except ValueError:
                pass

            # Add or init?
            if k in data:
                data[k].append(v)
            else:
                data[k] = [
                    v,
                ]

    return data


def load_spikes(filename):
    """Read in a spikes .csv
    
    NOTE: assumes data was written be write_spikes().
    """
    data = _read_csv_cols_into_dict(filename)
    if len(data) == 0:
        return np.array([]), np.array([])

    ns = np.asarray(data['ns'])
    ts = np.asarray(data['ts'])

    return ns, ts


def write_traces(name, results, select=None):
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
