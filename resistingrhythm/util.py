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


def locate_firsts(ns, ts, combine=False):
    # Recode neurons as coming from one neuron,
    # i.e. a hack to examine the network
    if combine:
        ns = np.zeros_like(ns)

    ns_first, ts_first = [], []
    for n in np.unique(ns):
        ns_n, ts_n = select_n(n, ns, ts)

        loc = np.argsort(ts_n).argmin()

        ns_first.append(ns_n[loc])
        ts_first.append(ts_n[loc])

    return np.asarray(ns_first), np.asarray(ts_first)


def select_voltages(budget, select=None):
    if select is None:
        select = ["V_m", "V_m_thresh", "V_comp", "V_osc", "V_free"]

    voltages = {}
    for k, v in budget.items():
        if k in select:
            voltages[k] = v

    return voltages


def locate_peaks(budget, onset=None, offset=None, combine=False, select=None):
    # Extract
    vm = budget["V_m"]
    times = budget['times']

    # Window?
    if onset is not None:
        m = np.logical_and(times > onset, times <= offset)
        vm = vm[:, m]
        times = times[m]

    # Create ns
    ns = np.arange(vm.shape[0])

    # Find ts
    idx = np.argmax(vm, axis=1)
    ts = []
    for i in idx:
        ts.append(times[i])
    ts = np.asarray(ts)

    if combine:
        ns = np.zeros(1)
        ts = np.asarray([np.min(ts)])

    return ns, ts


def budget_window(budget, t, budget_width, select=None):
    if budget_width < 0:
        raise ValueError("budget_width must be positive")

    # Disassemble budget into voltages and times
    times = np.squeeze(budget['times'])
    voltages = select_voltages(budget, select=select)

    # Filter from (t, t + budget_width)
    filtered = {}
    for k, v in voltages.items():
        if v.ndim == 2:
            t_on = t
            t_off = t_on + budget_width

            window = (t_on, t_off)
            m = np.logical_and(times > window[0], times < window[1])

            filtered[k] = budget[k][:, m]
            filtered['times'] = times[m]
        else:
            raise ValueError("{} must be 2d".format(k))

    return filtered


def filter_voltages(budget,
                    ns_first,
                    ts_first,
                    budget_delay=-4e-3,
                    budget_width=4e-3,
                    select=None,
                    combine=False):

    # Sanity
    if budget_width < 0:
        raise ValueError("budget width must be positive")
    if np.abs(budget_delay) < budget_width:
        raise ValueError("delay must be greater than width")

    # Disassemble budget into voltages and times
    times = np.squeeze(budget['times'])
    voltages = select_voltages(budget, select=select)

    # Filter based on first passage times
    filtered = {}
    for k, v in voltages.items():
        if v.ndim > 2:
            raise ValueError("{} is greater than 2d.".format(k))
        elif v.ndim == 2:
            if combine:
                t = ts_first[0]
                t_on = t + budget_delay
                t_off = t_on + budget_width

                window = (t_on, t_off)
                m = np.logical_and(times > window[0], times < window[1])

                filtered[k] = budget[k][:, m]
                filtered['times'] = times[m]

            else:
                xs = []
                x_times = []
                for i, n in enumerate(ns_first):
                    t = ts_first[i]
                    t_on = t + budget_delay
                    t_off = t_on + budget_width

                    window = (t_on, t_off)
                    m = np.logical_and(times > window[0], times < window[1])

                    xs.append(budget[k][n, m])
                    x_times.append(times[m])

                # Sometimes for baffling reasons xs,times are ragged.
                # Keep the shortest len
                min_l = np.min([len(x) for x in xs])

                xs_f = []
                for x in xs:
                    x_f = x[0:min_l]
                    xs_f.append(x_f)

                x_times_f = []
                for x in x_times:
                    x_t = x[0:min_l]
                    x_times_f.append(x_t)

                # Save, finally....
                filtered[k] = np.vstack(xs_f)
                filtered['times'] = np.vstack(x_times_f)

        else:
            raise ValueError("{} is less than 2d".format(k))

    return filtered


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


def find_E(E_0, ns_ref, ts_ref, no_lock=False, verbose=False):
    """Find the ref spike closest to E_0"""

    if no_lock:
        return E_0

    if np.isclose(E_0, 0.0):
        _, E = locate_firsts(ns_ref, ts_ref, combine=True)
        if verbose:
            print(">>> Locking on first spike. E was {}.".format(E))
    else:
        E = nearest_spike(ts_ref, E_0)
        if verbose:
            print(">>> E_0 was {}, using closest at {}.".format(E_0, E))

    return E


def find_phis(E, f, d, verbose=False):
    """Find the phase begin a osc cycle at (E + d)"""
    phi_E = float(-E * 2 * np.pi * f)
    phi_w = float((-(E + d) * 2 * np.pi * f) + np.pi / 2)

    if verbose:
        print(">>> phi_w {}, phi_E {}".format(phi_w, phi_E))

    return phi_w, phi_E


def score_by_group(ts_ref, ts_n):
    var = mad(ts_n)
    error = mae(ts_n, ts_ref)

    return var, error


def score_by_n(N, ns_ref, ts_ref, ns_n, ts_n):
    v_i = []
    e_i = []
    for i in range(N):
        ns_ref_i, ts_ref_i = select_n(i, ns_ref, ts_ref)
        ns_i, ts_i = select_n(i, ns_n, ts_n)

        # Variance
        v_i.append(mad(ts_i))

        # Error
        e_i.append(mae(ts_i, ts_ref_i))

    # Expectation of all neurons.
    var = np.mean(v_i)
    error = np.mean(e_i)

    return var, error


def estimate_communication(ns, ts, window, coincidence_t=1e-3, min_spikes=2):
    # Define overall analysis window
    t0 = window[0]
    tn = window[1]

    times = create_times(window, coincidence_t)
    n_steps = times.size

    # If there are not spikes there is not communication.
    if ns.size == 0:
        return 0

    m = np.logical_and(t0 <= ts, ts <= tn)
    ts = ts[m]
    ns = ns[m]

    # Calculate # coincidences C for every possible
    # coincidence (CC) window, for all time.
    C = 0
    for i in range(n_steps - 1):

        # Get CC window
        cc0 = times[i]
        ccn = times[i + 1]
        m = np.logical_and(cc0 <= ts, ts <= ccn)

        # Count spikes in the window
        if ts[m].size >= min_spikes:
            C += 1

    return C


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


def burst(tspan, onset, n_cycles, A, f, phi, dt, min_A=0.0):
    # Define time
    times = create_times(tspan, dt=dt)

    # Create a sin wave over the full tspan
    osc = A / 2 * (1 + np.sin((times * f * 2 * np.pi) + phi))

    # Truncate it to a n_cycle burst, starting at onset
    if np.isclose(f, 0.0):
        osc[:] = min_A
    else:
        burst_l = (1 / float(f)) * n_cycles

        m = np.logical_not(
            np.logical_and(times >= onset, times <= (onset + burst_l)))
        osc[m] = min_A

    return times, osc


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


def pulse(I, on, off, t, dt):
    times = create_times((0, 1), dt)
    wave = np.zeros_like(times)
    i = find_time_index(times, on)
    j = find_time_index(times, off)
    wave[i:j] = I

    return wave


def poisson_impulse(t, t_stim, w, rate, n=10, dt=1e-3, seed=None):
    """Create a pulse of spikes w seconds wide, starting at t_stim."""

    # Poisson sample the rate over w
    times = fsutil.create_times(t, dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_stim, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


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


def read_modes(mode, json_path=None):
    # Read in modes from the detault location
    # or with what the user provided?
    if json_path is None:
        json_path = os.path.join(
            os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    # Extract params
    params = modes[mode]

    # And default input
    initial_inputs = params.pop('initial_inputs')
    w_in = initial_inputs['w_in']
    bias_in = initial_inputs['bias_in']
    sigma = initial_inputs['sigma']

    return params, w_in, bias_in, sigma


def get_mode_names(json_path=None):
    """List all modes"""
    if json_path is None:
        json_path = os.path.join(
            os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    return modes.keys()


def get_default_modes():
    """Get the default modes built into voltagebudget library"""
    json_path = os.path.join(
        os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    return modes


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


def read_stim(stim):
    """Read in budget_experiment stimulation, as a dict"""
    return _read_csv_cols_into_dict(stim)


def read_results(results):
    """Read in budget_experiment results, as a dict"""
    return _read_csv_cols_into_dict(results)


def read_args(args):
    """Read in an adex arguments file, as a dict"""
    reader = csv.reader(open(args, 'r'))
    args_data = {}
    for row in reader:
        k = row[0]
        v = row[1]

        # Convert numbers
        if k == 'seed':
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass

        # Convert bools
        if v.strip() == 'True':
            v = True
        if v.strip() == 'False':
            v = False

        # save
        args_data[k] = v

    return args_data
