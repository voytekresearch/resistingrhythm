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
from resistingrhythm.util import l1_by_n
from resistingrhythm.util import l2_by_n


def run(run_name,
        stim_name=None,
        osc_name=None,
        t=12,
        burn_t=10,
        N=10,
        Ca_target=0.03,
        tau_h=1,
        w_min=5e-6,
        w_max=50e-6,
        sigma=0,
        num_trials=2,
        seed_value=42,
        verbose=False):
    """Run a HHH experiment"""

    # ---------------------------------------------------
    # Fixed params
    bias_in = 0
    time_step = 1e-5

    # ---------------------------------------------------
    if verbose:
        print(">>> Running {}".format(run_name))
        print(">>> Checking args.")

    N = int(N)
    num_trials = int(num_trials)

    t = float(t)
    w_min = float(w_min)
    w_max = float(w_max)
    tau_h = float(tau_h)
    Ca_target = float(Ca_target)
    burn_t = float(burn_t)

    if w_min > w_max:
        raise ValueError("w_min must be smaller than w_max")
    if t <= 0:
        raise ValueError("t must be positive")
    if burn_t >= t:
        raise ValueError("burn_t must be less than t")
    if N < 1:
        raise ValueError("N must be positive")
    if tau_h < 1:
        raise ValueError("tau_h must be greater than 1")
    if num_trials < 1:
        raise ValueError("num_trials must be greater than 1")

    # -
    a_window = (burn_t + 1e-3, t)
    w = (w_min, w_max)

    if verbose:
        print(">>> Analysis window: {}".format(a_window))
        print(">>> Weight range: {}".format(w))

    # ---------------------------------------------------
    # load spikes
    ns_stim, ts_stim = load_spikes(stim_name)
    ns_osc, ts_osc = load_spikes(osc_name)
    if verbose:
        print(">>> Loaded {}".format(stim_name))
        print(">>> Loaded {}".format(osc_name))

    # ---------------------------------------------------
    results = []
    for k in range(num_trials):
        if verbose:
            print(">>> Running trial {}".format(k))
            print(">>> Running the reference model")

        # No osc
        results_ref = HHH(
            t,
            ns_stim,
            ts_stim,
            np.asarray([]),  # no osc mod
            np.asarray([]),
            N=N,
            Ca_target=Ca_target,
            bias_in=bias_in,
            sigma=sigma,
            w_in=w,
            w_osc=w,
            tau_h=tau_h,
            time_step=time_step,
            burn_time=burn_t,
            seed_value=seed_value + k,
            record_traces=False,
            homeostasis=True)

        ns_ref, ts_ref = filter_spikes(results_ref["ns"], results_ref["ts"],
                                       a_window)

        # -
        if verbose:
            print(">>> Running the modulation model")
        results_k = HHH(t,
                        ns_stim,
                        ts_stim,
                        ns_osc,
                        ts_osc,
                        N=N,
                        Ca_target=Ca_target,
                        bias_in=bias_in,
                        sigma=sigma,
                        w_in=w,
                        w_osc=w,
                        tau_h=tau_h,
                        time_step=time_step,
                        burn_time=burn_t,
                        seed_value=seed_value + k,
                        record_traces=True,
                        homeostasis=True)

        # Select spikes in a_window
        ns_k, ts_k = filter_spikes(results_k['ns'], results_k['ts'], a_window)

        # Analysis of error and network corr, in a_window
        k_error = kappa(ns_ref, ts_ref, ns_k, ts_k, a_window, dt=time_step)
        k_coord = kappa(ns_k, ts_k, ns_k, ts_k, a_window, dt=time_step)

        # l1 scores
        abs_var, abs_error = l1_by_n(N, ns_ref, ts_ref, ns_k, ts_k)

        # l2 scores
        mse_var, mse_error = l2_by_n(N, ns_ref, ts_ref, ns_k, ts_k)

        # Get rate
        rate_k = float(ns_k.size) / float(N) / (t - burn_t)

        # Get final avg. calcium
        # over a window? Just grab last?
        Ca_obs = results_k['calcium'][:, -50].mean()

        # Get final avg V_m
        V_m = results_eq['v_m'][:, -50].mean()

        # save row
        row = (i, V_m, Ca_eq, Ca_k, Ca_obs, error, coord, abs_error, abs_var,
               mse_error, mse_var, rate_k)
        results.append(row)

    # ---------------------------------------------------
    if verbose:
        print(">>> Saving results")

    head = [
        "i", "V_m", "Ca_eq", "Ca_target", "Ca_obs", "kappa_error",
        "kappa_coord", "abs_error", "abs_var", "mse_error", "mse_var", "rate"
    ]
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