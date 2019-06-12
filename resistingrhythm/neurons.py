import inspect
import numpy as np

from brian2 import *
from copy import deepcopy


def _parse_membrane_param(x, n, prng):
    try:
        if len(x) == 2:
            x_min, x_max = x
            x = prng.uniform(x_min, x_max, n)
        else:
            raise ValueError("Parameters must be scalars, or 2 V_lement lists")
    except TypeError:
        pass

    return x, prng


def HHH(time,
        ns_in,
        ts_in,
        ns_osc,
        ts_osc,
        external_current=None,
        Ca_target=50e-9,
        tau_h=10,
        g_l=1.0e-3,
        g_Ca=0.03e-3,
        G_Ca=60e-3,
        G_KCa=60e-3,
        sigma=0,
        N=1,
        w_in=0.8e-9,
        tau_in=5e-3,
        V_in=0,
        bias_in=0.0e-9,
        w_osc=0.8e-9,
        tau_osc=5e-3,
        V_osc=0,
        p_connection=0.1,
        burn_time=0,
        time_step=1e-5,
        homeostasis=True,
        record_traces=True,
        progress_report=None,
        seed_value=None):
    """Homeostasis in Hippocampal HH neurons."""
    prefs.codegen.target = 'numpy'

    prng = np.random.RandomState(seed_value)
    seed(seed_value)

    time_step *= second
    defaultclock.dt = time_step

    # ----------------------------------------------------
    # User set (in SI units)
    bias_in *= amp

    # Input constants
    # w_in/w_osc are parsed and given units below...
    tau_in *= second
    V_in *= volt

    w_osc *= siemens
    tau_osc *= second
    V_osc *= volt

    # Noise scale
    # sigma *= siemens

    # While concentration vars are said to be
    # passed in 'molar',
    # we correct for the way units in brian2
    # define liter and how that interacts with
    # the float() values of numbers who've been
    # touched by volume units.
    # For example, if you passed 1 and we convert to
    # 1 liter internally, then cast back to float the
    # '1' becomes 0.001.
    # that is
    # > float(1 * liter)
    # 0.001
    # **To correct for that, we use mmolar in all our unit conversions.**
    Ca_target *= mmolar
    Ca = Ca_target
    tau_h *= second

    # ----------------------------------------------------
    # HH general params, in misc units
    Et = 20 * mvolt
    Cm = 1 * uF  # /cm2
    g_l *= siemens

    V_K = -100 * mV  # was 100, changed to match LeMasson
    V_l = -70 * mV
    V_Na = 50 * mV

    # Ca and Homeostasis values from
    # Siegel, M., Marder, & Abbott, L.F., 1994. Activity-dependent
    # current distributions in model neurons. 91, pp.11308-11312.

    # d[Ca]/dt
    delta = 0.6 * umolar  # TODO: was umolar?
    k = 1 / (200.0 * msecond)
    gamma = -4.7e-2 * (mmolar / mamp / msecond)
    V_Ca = 150 * mV
    V1 = -50 * mV
    V2 = 10 * mV
    g_Ca *= siemens
    G_Ca *= siemens

    # dg/dt
    G_Na = 360 * msiemens
    G_K = 120 * msiemens  # Try 90?
    G_KCa *= siemens

    g_Na = G_Na / 2  # Init
    g_K = G_K / 2
    g_KCa = G_KCa / 2

    # ----------------------------------------------------
    eqs = """
    dV/dt = (I_Na + I_K + I_KCa + I_Ca + I_l + bias_in + I_noi + I_in + I_osc + I_ext) / Cm : volt
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
    I_KCa = g_KCa * (m_KCa ** 4) * (V_K - V) : amp
    dm_KCa/dt = (m_KCa_inf - m_KCa) / tau_m_KCa : 1
    m_KCa_inf = (Ca / (Ca + 3 * molar)) * (1 / (1 + exp((V/mV + 28.3) / -12.6))) : 1
    tau_m_KCa = (90.3 - (75.1 / (1 + exp((V/mV + 46) / -22.7)))) * ms : second
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_noi = g_noi * (V_l - V) : amp
    dg_noi/dt = (-g_noi / tau_in) + (sqrt(sigma / tau_in) * xi) * (siemens) : siemens
    """ + """
    I_Ca = -g_Ca * (1 + tanh((V - V1) / V2)) * (V - V_Ca): amp
    dCa/dt = (-k * Ca) - (gamma * I_Ca) : mmolar
    """ + """
    g_total = g_in + g_osc : siemens
    I_in = g_in * (V_in - V) : amp
    I_osc = g_osc * (V_osc - V) : amp
    dg_in/dt = -g_in / tau_in : siemens
    dg_osc/dt = -g_osc / tau_osc : siemens
    """ + """
    Ca_target : mmolar
    # g_KCa : siemens
    """

    if homeostasis:
        eqs += """ 
        dg_Na/dt = (1 / tau_h) * (G_Na / (1 + exp(1 * (Ca - Ca_target)/delta)) - g_Na) : siemens 
        dg_Ca/dt = (1 / tau_h) * (G_Ca / (1 + exp(1 * (Ca - Ca_target)/delta)) - g_Ca) : siemens 
        # g_Ca : siemens
        dg_K/dt = (1 / tau_h) * (G_K / (1 + exp(-1 * (Ca - Ca_target)/delta)) - g_K) : siemens 
        dg_KCa/dt = (1 / tau_h) * (G_KCa / (1 + exp(-1 * (Ca - Ca_target)/delta)) - g_KCa) : siemens 
        """
    else:
        eqs += """
        g_Na : siemens
        g_K : siemens
        g_Ca : siemens
        g_KCa : siemens
        """

    # Setup the current
    if external_current is not None:
        I_x = TimedArray(external_current, dt=time_step)
        eqs += """I_ext = I_x(t) * amp : amp"""
    else:
        eqs += """I_ext = 0 * amp : amp"""

    # ----------------------------------------------------
    # Def the net by hand....
    net = Network()
    to_monitor = [
        'V', 'g_total', 'g_Na', 'g_Ca', 'g_K', 'g_KCa', 'Ca', 'I_Ca', 'I_Na',
        'I_K'
    ]

    # -
    # The target pop....
    P_target = NeuronGroup(
        N, eqs, threshold='V > Et', refractory=2 * ms, method='euler')

    P_target.V = V_l
    P_target.g_Na = g_Na
    P_target.g_K = g_K
    P_target.g_KCa = g_KCa
    P_target.g_Ca = g_Ca
    P_target.Ca_target = Ca_target
    P_target.Ca = Ca_target

    net.add(P_target)

    # -
    # Connect in
    if ns_in.size > 0:
        # Make a population out of spiking input
        P_in = SpikeGeneratorGroup(np.max(ns_in) + 1, ns_in, ts_in * second)

        # Connect to P_target
        C_in = Synapses(
            P_in, P_target, model='w_in : siemens', on_pre='g_in += w_in')

        C_in.connect(p=p_connection)

        # Finally, set potentially random weights
        w_in, prng = _parse_membrane_param(w_in, len(C_in), prng)
        C_in.w_in = w_in * siemens

        net.add([P_in, C_in])
        to_monitor.append('I_in')

    # -
    # Connect osc
    if ns_osc.size > 0:
        # Make a population out of spiking input
        P_osc = SpikeGeneratorGroup(
            np.max(ns_osc) + 1, ns_osc, ts_osc * second)

        # Connect to P_target
        C_osc = Synapses(
            P_osc, P_target, model='w_osc : siemens', on_pre='g_osc += w_osc')

        C_osc.connect(p=p_connection)

        # Finally, set potentially random weights
        w_osc, prng = _parse_membrane_param(w_osc, len(C_osc), prng)
        C_osc.w_osc = w_osc * siemens

        net.add([P_osc, C_osc])
        to_monitor.append('I_osc')

    # -
    # Setup recording, but don't add it to the net yet....
    spikes = SpikeMonitor(P_target)
    if record_traces:
        traces = StateMonitor(P_target, to_monitor, record=True)

    # ----------------------------------------------------
    # !
    if burn_time > 0:
        if burn_time <= 0:
            raise ValueError("burn_time must be <= time")

        # Run without recording.
        net.run(burn_time * second, report=progress_report)

        # Start recording
        if record_traces:
            net.add(traces)
        net.add(spikes)

        net.run((time - burn_time) * second, report=progress_report)
    else:
        if record_traces:
            net.add(traces)
        net.add(spikes)

        net.run(time * second, report=progress_report)

    # ----------------------------------------------------
    # Unpack the results
    ns, ts = np.asarray(spikes.i_), np.asarray(spikes.t_)

    if record_traces:
        times = np.asarray(traces.t_)
        vm = np.asarray(traces.V_)
        g_total = np.asarray(traces.g_total_)
        g_Na = np.asarray(traces.g_Na_)
        g_K = np.asarray(traces.g_K_)
        g_KCa = np.asarray(traces.g_KCa_)
        g_Ca = np.asarray(traces.g_Ca_)
        I_Ca = np.asarray(traces.I_Ca_)
        I_Na = np.asarray(traces.I_Na_)
        I_K = np.asarray(traces.I_K_)
        calcium = np.asarray(traces.Ca)

        # and repack them
        results = {
            'ns': ns,
            'ts': ts,
            'times': times,
            'v_m': vm,
            'g_total': g_total,
            'calcium': calcium,
            'g_Ca': g_Ca,
            'g_KCa': g_KCa,
            'g_Na': g_Na,
            'I_Ca': I_Ca,
            'I_Na': I_Na,
            'I_K': I_K,
            'g_K': g_K
        }
        if ns_osc.size > 0:
            results["I_osc"] = np.asarray(traces.I_osc_)
        if ns_in.size > 0:
            results["I_in"] = np.asarray(traces.I_in_)

    else:
        results = {'ns': ns, 'ts': ts}

    return results
