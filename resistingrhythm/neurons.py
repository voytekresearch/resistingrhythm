import inspect
import numpy as np

from brian2 import *
from copy import deepcopy


def HHH(time,
        ns_in,
        ts_in,
        ns_osc,
        ts_osc,
        Ca_target=50e-9,
        tau_h=10,
        N=1,
        w_in=0.8e-9,
        tau_in=5e-3,
        V_in=0,
        bias_in=0.0e-9,
        w_osc=0.8e-9,
        tau_osc=5e-3,
        V_osc=0,
        sigma=0,
        time_step=1e-5,
        report=None,
        seed_value=None):
    """Homeostasis in HH neurons."""
    prefs.codegen.target = 'numpy'
    seed(seed_value)

    time_step *= second
    defaultclock.dt = time_step

    # ----------------------------------------------------
    # User set (in SI units)
    bias_in *= amp

    # Input constants
    w_in *= siemens
    tau_in *= second
    V_in *= volt

    w_osc *= siemens
    tau_osc *= second
    V_osc *= volt

    # Noise scale
    sigma *= siemens

    # The homeo. eq point and tau
    Ca_target *= molar
    tau_h *= second

    # ----------------------------------------------------
    # HH general params, in misc units
    Et = 20 * mvolt
    Cm = 1 * uF  # /cm2

    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_A = 80 * msiemens
    g_l = 0.1 * msiemens

    V_Na = 50 * mV
    V_K = -100 * mV  # was 100, changed to match LeMasson
    V_l = -67 * mV
    V_A = -80 * mV

    # Ca + homeo specific
    delta = 0.6 * umolar
    Ca = 5e-5 * mmolar
    k = 1 / (600.0 * msecond)
    gamma = -4.7e-2 * (mmolar / mamp / msecond)

    V_Ca = 120 * mV
    V1 = -50 * mV
    V2 = 10 * mV
    g_Ca = 0.03 * msiemens

    G_Na = 360 * msiemens
    G_K = 180 * msiemens

    # ----------------------------------------------------
    hh = """
    dV/dt = (I_Na + I_K + I_A + I_l + bias_in + I_noi + I_in + I_osc) / Cm : volt
    """ + """
    I_Na = g_Na * (m ** 3) * h * (V_Na - V) : amp
    m = a_m / (a_m + b_m) : 1
    a_m = (0.32 * (54 + V/mV)) / (1 - exp(-0.25 * (V/mV + 54))) / ms : Hz
    b_m = (0.28 * (27 + V/mV)) / (exp(0.2 * (V/mV + 27)) - 1) / ms : Hz
    h = clip(1 - 1.25 * n, 0, inf) : 1
    """ + """
    I_K = g_K * n ** 4 * (V_K - V) : amp
    dn/dt = (a_n - (a_n * n)) - b_n * n : 1
    a_n = (0.032 * (52 + V/mV)) / (1 - exp(-0.2 * (V/mV + 52))) / ms : Hz
    b_n = 0.5 * exp(-0.025 * (57 + V/mV)) / ms : Hz
    """ + """
    I_A = g_A * (m_A ** 3) * h_A * (V_A - V) : amp
    dm_A/dt = (m_A_inf - m_A) / tau_m_A : 1
    dh_A/dt = (h_A_inf - h_A) / tau_h_A : 1
    m_A_inf = 1 / (1 + exp((V/mV + 27.2) / -8.7)) : 1
    h_A_inf = 1 / (1 + exp((V/mV + 56.9) / 4.9)) : 1
    tau_m_A = (11.6 - (10.4 / (1 + exp((V/mV + 32.9) / -15.2)))) * ms : second
    tau_h_A = (36.8 - (29.2 / (1 + exp((V/mV + 38.9) / -26.5)))) * ms : second
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_noi = g_noi * (V_l - V) : amp
    dg_noi/dt = -(g_noi + (sigma * sqrt(tau_in) * xi)) / tau_in : siemens
    """ + """
    I_Ca = -g_Ca * (1 + tanh((V - V1) / V2)) * (V - V_Ca): amp
    dCa/dt = (-k * Ca) - (gamma * I_Ca) : mmolar
    """ + """
    dg_Na/dt = (1 / tau_h) * (G_Na / (1 + exp(1 * (Ca - Ca_target)/delta)) - g_Na) : siemens 
    dg_K/dt = (1 / tau_h) * (G_K / (1 + exp(-1 * (Ca - Ca_target)/delta)) - g_K) : siemens 
    """ + """
    g_total = g_in + g_osc : siemens
    I_in = g_in * (V_in - V) : amp
    I_osc = g_osc * (V_osc - V) : amp
    dg_in/dt = -g_in / tau_in : siemens
    dg_osc/dt = -g_osc / tau_osc : siemens
    """

    # ----------------------------------------------------
    # Def the net by hand....
    net = Network()

    # -
    # The target pop....
    P_target = NeuronGroup(
        N, hh, threshold='V > Et', refractory=2 * ms, method='euler')

    P_target.V = V_l
    P_target.g_Na = g_Na
    P_target.g_K = g_K
    P_target.Ca = Ca

    net.add(P_target)

    # -
    # Connect in
    if ns_in.size > 0:
        P_in = SpikeGeneratorGroup(np.max(ns_in) + 1, ns_in, ts_in * second)

        C_in = Synapses(
            P_in, P_target, model='w_in : siemens', on_pre='g_in += w_in')
        C_in.connect()

        C_in.w_in = w_in

        net.add([P_in, C_in])

    # -
    # Connect osc
    if ns_osc.size > 0:
        P_osc = SpikeGeneratorGroup(
            np.max(ns_osc) + 1, ns_osc, ts_osc * second)

        C_osc = Synapses(
            P_osc, P_target, model='w_osc : siemens', on_pre='g_osc += w_osc')
        C_osc.connect()

        C_osc.w_osc = w_osc

        net.add([P_osc, C_osc])

    # -
    # Data acq
    spikes = SpikeMonitor(P_target)

    # TODO add Ca dynamics
    # to_monitor = ['V', 'g_total']
    to_monitor = ['V', 'g_total', 'g_Na', 'g_K', 'Ca']
    traces = StateMonitor(P_target, to_monitor, record=True)

    net.add([spikes, traces])

    # !
    net.run(time * second, report=report)

    # ----------------------------------------------------
    # Unpack the results
    ns, ts = np.asarray(spikes.i_), np.asarray(spikes.t_)

    times = np.asarray(traces.t_)
    vm = np.asarray(traces.V_)
    g_total = np.asarray(traces.g_total_)
    g_Na = np.asarray(traces.g_Na_)
    g_K = np.asarray(traces.g_K_)
    calcium = np.asarray(traces.Ca)

    # and repack them
    results = {
        'ns': ns,
        'ts': ts,
        'times': times,
        'v_m': vm,
        'g_total': g_total,
        'calcium': calcium,
        'g_Na': g_Na,
        'g_K': g_K
    }

    return results


def homeostadex(N,
                time,
                ns,
                ts,
                V_set,
                E=0,
                n_cycles=1,
                w_in=0.8e-9,
                tau_in=5e-3,
                bias_in=0.0e-9,
                V_t=-50.0e-3,
                V_thresh=0.0,
                f=0,
                A=.1e-9,
                phi=0,
                sigma=0,
                C=200e-12,
                g_l=10e-9,
                V_l=-70e-3,
                a=0e-9,
                b=10e-12,
                tau_w=30e-3,
                V_rheo=-48e-3,
                delta_t=2e-3,
                time_step=1e-5,
                budget=True,
                report=None,
                save_args=None,
                pulse_params=None,
                seed_value=42):
    """A AdEx neuron
    
    Params
    ------
    time : Numeric
        Simulation run time (seconds)

    [...]

    pulse_params: None or a tuple: (I, on, off)
        Inject a current I, starting at on, ending on off
    seed : None, int
        The random seed
    """
    # -----------------------------------------------------------------
    # Plant all the seeds!
    seed(seed_value)
    prng = np.random.RandomState(seed_value)

    # Integration settings
    defaultclock.dt = time_step * second
    prefs.codegen.target = 'numpy'

    # -----------------------------------------------------------------
    if save_args is not None:
        skip = ['ns', 'ts', 'save_args']
        arg_names = inspect.getargspec(adex)[0]

        args = []
        for arg in arg_names:
            if arg not in skip:
                row = (arg, eval(arg))
                args.append(row)

        with open("{}.csv".format(save_args), "w") as fi:
            writer = csv.writer(fi, delimiter=",")
            writer.writerows(args)

    # -----------------------------------------------------------------
    # If there's no input, return empty
    if (ns.shape[0] == 0) and (pulse_params is None):
        return np.array([]), np.array([]), dict()

    # -----------------------------------------------------------------
    # Adex dynamics params
    g_l, prng = _parse_membrane_param(g_l, N, prng)
    V_l, prng = _parse_membrane_param(V_l, N, prng)
    C, prng = _parse_membrane_param(C, N, prng)

    # Potentially random synaptic params
    # Note: w_in gets created after synaptic input is 
    # Defined.
    bias_in, prng = _parse_membrane_param(bias_in, N, prng)
    tau_in, prng = _parse_membrane_param(tau_in, N, prng)

    # Potentially random membrane params
    V_rheo, prng = _parse_membrane_param(V_rheo, N, prng)
    a, prng = _parse_membrane_param(a, N, prng)
    b, prng = _parse_membrane_param(b, N, prng)
    delta_t, prng = _parse_membrane_param(delta_t, N, prng)
    tau_w, prng = _parse_membrane_param(tau_w, N, prng)

    # Fixed membrane dynamics
    sigma *= siemens
    V_cut = V_t + 8 * np.mean(delta_t)
    V_thresh *= volt

    # Oscillation params
    f *= Hz
    A *= amp
    phi *= second

    # -----------------------------------------------------------------
    # Define an adex neuron, and its connections
    eqs = """
    dv/dt = (-g_l * (v - V_l) + (g_l * delta_t * exp((v - V_t) / delta_t)) + I_in + I_osc(t) + I_noise + I_ext + bias_in - w) / C : volt
    dw/dt = (a * (v - V_l) - w) / tau_w : amp
    dh/dt = (c * (v - V_set) - h) / tau_h : amp
    dg_in/dt = -g_in / tau_in : siemens
    dg_noise/dt = -(g_noise + (sigma * sqrt(tau_in) * xi)) / tau_in : siemens
    I_in = g_in * (v - V_l) : amp
    I_noise = g_noise * (v - V_l) : amp
    C : farad
    g_l : siemens 
    a : siemens
    b : amp
    delta_t : volt
    tau_w : second
    V_rheo : volt
    V_l : volt
    bias_in : amp
    tau_in : second
    """

    # Add step?
    # A step of current injection?code clean
    if pulse_params is not None:
        I, t_on, t_off = pulse_params
        waves = pulse(I, t_on, t_off, time, time_step)
        I_sq = TimedArray(waves, dt=time_step * second)
        eqs += """I_ext = I_sq(t) * amp : amp"""
    else:
        eqs += """I_ext = 0 * amp : amp"""

    # Create osc/burst
    if np.isclose(E, 0.0):
        E = time

    _, I_osc = burst((0, time), E, n_cycles, A,
                     float(f), float(phi), float(time_step))
    I_osc = TimedArray(I_osc, dt=time_step * second)

    # Def the population
    P_n = NeuronGroup(
        N,
        model=eqs,
        threshold='v > V_thresh',
        reset="v = V_rheo; w += b",
        method='euler')

    # Init adex params
    # Fixed voltages neuron params
    V_t *= volt
    V_cut *= volt

    P_n.a = a * siemens
    P_n.b = b * amp
    P_n.delta_t = delta_t * volt
    P_n.tau_w = tau_w * second
    P_n.V_rheo = V_rheo * volt
    P_n.C = C * farad
    P_n.g_l = g_l * siemens
    P_n.V_l = V_l * volt
    P_n.bias_in = bias_in * amp
    P_n.tau_in = tau_in * second

    # Init V0, w0
    V_rest = V_l + (bias_in / g_l)
    P_n.v = V_rheo * volt
    P_n.w = 0 * pamp

    # -----------------------------------------------------------------
    # Add synaptic input into the network.
    if ns.size > 0:
        P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
        C_stim = Synapses(
            P_stim, P_n, model='w_in : siemens', on_pre='g_in += w_in')
        C_stim.connect()

        # (Finally) Potentially random weights
        w_in, prng = _parse_membrane_param(w_in, len(C_stim), prng)
        C_stim.w_in = w_in * siemens

    # -----------------------------------------------------------------
    # Record input and voltage 
    spikes_n = SpikeMonitor(P_n)

    record = ['v', 'I_ext']
    traces_n = StateMonitor(P_n, record, record=True)

    # -----------------------------------------------------------------
    # Build the model!
    net = Network(P_n, traces_n, spikes_n)
    net.store('no_stim')

    # -
    # Run the net without any stimulation. 
    # (This strictly speaking isn't
    # necessary, but I can't get Brian to express the needed
    # diff eq to get the osc budget term in one pass...)
    net.run(time * second, report=report)
    V_osc = deepcopy(np.asarray(traces_n.v_))

    # -
    # Run with stim
    net.restore('no_stim')

    # Add spikes?
    if ns.size > 0:
        net.add([P_stim, C_stim])

    net.run(time * second, report=report)

    # -----------------------------------------------------------------
    # Extract data from the run model
    # Spikes
    ns_e = np.asarray(spikes_n.i_)
    ts_e = np.asarray(spikes_n.t_)

    # Define the return objects
    result = [ns_e, ts_e]

    # Define the terms that go into the budget
    # (these get added to result at the end)
    if budget:
        # -
        # Drop brian units...
        V_cut = float(V_cut)
        V_t = float(V_t)
        V_thresh = float(V_thresh)

        V_rheo = np.asarray(V_rheo)
        V_leak = np.asarray(V_l)

        # -
        # Get Vm
        V_m = np.asarray(traces_n.v_)

        # -
        # Rectify away spikes
        # V_m
        V_m_thresh = V_m.copy()
        if V_rheo.size > 1:
            # V_m 
            rect_mask = V_m_thresh > V_rheo[:, None]
            for i in range(V_m_thresh.shape[0]):
                V_m_thresh[i, rect_mask[i, :]] = V_rheo[i]

            # and V_osc
            rect_mask = V_osc > V_rheo[:, None]
            for i in range(V_osc.shape[0]):
                V_osc[i, rect_mask[i, :]] = V_rheo[i]
        else:
            # V_m and V_osc
            V_m_thresh[V_m_thresh > V_rheo] = V_rheo
            V_osc[V_osc > V_rheo] = V_rheo

        # Est. Comp AFTER spikes have been removed
        V_comp = V_osc - V_m_thresh
        V_comp[V_comp > 0] = 0  # Nonsense < 0 values creep in. Drop 'em.

        # Recenter V_osc so unit scale matches comp
        if V_leak.size > 1:
            V_osc = V_leak[:, None] - V_osc
        else:
            V_osc = V_leak - V_osc

        # Est free.
        if V_rheo.size > 1:
            V_free = V_rheo[:, None] - V_m_thresh
        else:
            V_free = V_rheo - V_m_thresh

        # Budget
        V_rest = np.asarray(V_rest)
        V_budget = V_rheo - V_rest

        # -
        # Build final budget for return
        vs = dict(
            tau_m=np.asarray(C / g_l),
            times=np.asarray(traces_n.t_),
            I_ext=np.asarray(traces_n.I_ext_),
            V_budget=V_budget,
            V_m=V_m,
            V_m_thresh=V_m_thresh,
            V_comp=V_comp,
            V_osc=V_osc,
            V_free=V_free,
            V_rheo=V_rheo,
            V_rest=V_rest,
            V_leak=V_leak,
            V_cut=V_cut,
            V_thresh=V_thresh,
            V_t=V_t)

        # -
        # Add the budget to the return var, result
        result.append(vs)

    return result
