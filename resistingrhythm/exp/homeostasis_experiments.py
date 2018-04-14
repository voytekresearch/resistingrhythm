import fire
import os
import csv

import numpy as np

from fakespikes.util import kappa

from resistingrhythm.neurons import HHH
from resistingrhythm.util import filter_spikes
from resistingrhythm.util import poisson_impulse
from resistingrhythm.util import poisson_oscillation
from resistingrhythm.util import load_spikes


def run(run_name,
        stim_name=None,
        osc_name=None,
        min_change=0.01,
        max_change=0.1,
        w_min=0.2e-6,
        w_max=2.0e-6,
        bias_in=100e-9,
        sigma=0,
        t=12,
        t_analysis_on=11,
        t_analysis_off=12,
        N=10,
        V_osc=0e-3,
        tau_h=5,
        n_samples=10,
        verbose=False):
    """Run a HHH experiment"""

    # ---------------------------------------------------
    # Arg sanity?
    N = int(N)
    n_samples = int(n_samples)

    t = float(t)
    w_min = float(w_min)
    w_max = float(w_max)
    tau_h = float(tau_h)
    V_osc = float(V_osc)
    percent_change = float(percent_change)
    t_analysis_on = float(t_analysis_on)
    t_analysis_off = float(t_analysis_off)

    if w_min > w_max:
        raise ValueError("w_min must be smaller than w_max")
    if t_analysis_off >= t_analysis_on:
        raise ValueError("t_analysis_off must be bigger than t_analysis_on")
    if t <= 0:
        raise ValueError("t must be positive")
    if t_analysis_off >= t:
        raise ValueError("t_analysis_off must be less than t")
    if N < 1:
        raise ValueError("N must be positive")
    if tau_h < 1:
        raise ValueError("tau_h must be greater than 1")
    if n_samples < 1:
        raise ValueError("n_samples must be greater than 1")
    if abs(percent_change) > 1:
        raise ValueError("percent_change must be <= to 1")

    # -
    a_window = (t_analysis_on, t_analysis_off)
    w = (w_min, w_max)

    # ---------------------------------------------------
    # load spikes
    ns_stim, ts_stim = load_spikes(stim_name)
    ns_osc, ts_osc = load_spikes(osc_name)

    # ---------------------------------------------------
    # Estimate the Ca_eq value.
    results_eq = HHH(
        t,
        np.asarray([]),  # No input 
        np.asarray([]),
        np.asarray([]),
        np.asarray([]),
        N=1,  # With no input, there is no need to run N > 1
        Ca=0,  # Let the system find its Ca value
        Ca_target=0,
        bias_in=bias_in,
        sigma=sigma,
        tau_h=tau_h,
        time_step=time_step,
        seed_value=seed_value,
        homeostasis=True)

    # To est Ca)eq, avg the last 50 times steps
    Ca_eq = results_eq['calcium'][:, -50].mean()

    # ---------------------------------------------------
    # run ref Ca_target = Ca_eq, osc = None
    # save spikes, no traces
    results_ref = HHH(
        t,
        ns_stim,
        ts_stim,
        np.asarray([]),  # no osc mod
        np.asarray([]),
        N=N,
        Ca=Ca_eq,
        Ca_target=Ca_eq + (Ca_eq * min_change),
        bias_in=bias_in,
        sigma=sigma,
        V_osc=V_osc,
        w_in=w,
        w_osc=w,
        tau_h=tau_h,
        time_step=time_step,
        seed_value=seed_value,
        record_traces=False,
        homeostasis=True)

    ns_ref = results_ref["ns"]
    ts_ref = results_ref["ts"]

    # ---------------------------------------------------
    # Pre-filter ref times
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, a_window)

    # A range of targets
    Ca_targets = np.linspace(Ca_eq + (Ca_eq * min_change),
                             Ca_eq + (Ca_eq * max_change), n_samples)

    results = []
    for i, Ca_t in enumerate(Ca_targets):
        # !
        results_t = HHH(t,
                        ns_stim,
                        ts_stim,
                        ns_osc,
                        ts_osc,
                        N=N,
                        Ca=Ca_eq,
                        Ca_target=Ca_t,
                        bias_in=bias_in,
                        sigma=sigma,
                        V_osc=V_osc,
                        w_in=w,
                        w_osc=w,
                        tau_h=tau_h,
                        time_step=time_step,
                        seed_value=seed_value,
                        record_traces=True,
                        homeostasis=True)

        # Select spikes in a_window
        ns_t, ts_t = filter_spikes(results_t['ns'], results_t['ts'], a_window)

        # Analysis of error and network corr, in a_window
        error = kappa(
            ns_ref,
            ts_ref,
            ns_t,
            ts_t, (t_analysis_on, t_analysis_off),
            dt=time_step)

        coord = kappa(
            ns_t,
            ts_t,
            ns_t,
            ts_t, (t_analysis_on, t_analysis_off),
            dt=time_step)

        # Get final avg. calcium
        # over a window? Just grab last?
        Ca_obs = results_t['calcium'][:, -50].mean()

        # Get final avg V_m
        V_m = results_eq['v_m'][:, -50].mean()

        # save row
        row = (i, V_m, Ca_eq, Ca_t, Ca_obs, error, coord)
        results.append(row)

    # ---------------------------------------------------
    # Write results to a .csv
    head = ["i", "V_m", "Ca_eq", "Ca_target", "Ca_obs", "error", "coord"]
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(head)
        writer.writerows(results)

    return None


if __name__ == "__main__":
    fire.Fire({
        'create_stim': poisson_impulse,
        'create_osc': poisson_oscillation,
        'run': run
    })