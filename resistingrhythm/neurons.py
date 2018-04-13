import inspect
import numpy as np

from brian2 import *
from copy import deepcopy


def HHH(time,
        ns_in,
        ts_in,
        ns_osc,
        ts_osc,
        Ca=50e-9,
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
    Ca *= mmolar
    tau_h *= second

    # ----------------------------------------------------
    # HH general params, in misc units
    Et = 20 * mvolt
    Cm = 1 * uF  # /cm2

    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_l = 0.1 * msiemens

    V_K = -100 * mV  # was 100, changed to match LeMasson
    V_l = -67 * mV
    V_Na = 50 * mV

    # Ca + homeo specific
    delta = 0.6 * umolar  # TODO: was umolar?
    k = 1 / (600.0 * msecond)
    gamma = -4.7e-2 * (mmolar / mamp / msecond)

    V_Ca = 150 * mV
    V1 = -50 * mV
    V2 = 10 * mV
    g_Ca = 0.03 * msiemens

    G_Na = 360 * msiemens
    G_K = 180 * msiemens

    # ----------------------------------------------------
    hh = """
    dV/dt = (I_Na + I_K + I_l + bias_in + I_noi + I_in + I_osc) / Cm : volt
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
    """ + """
    Ca_target : mmolar
    """

    # A

    # g_A = 80 * msiemens
    # V_A = -80 * mV
    #
    # I_A = g_A * (m_A ** 3) * h_A * (V_A - V) : amp
    # dm_A/dt = (m_A_inf - m_A) / tau_m_A : 1
    # dh_A/dt = (h_A_inf - h_A) / tau_h_A : 1
    # m_A_inf = 1 / (1 + exp((V/mV + 27.2) / -8.7)) : 1
    # h_A_inf = 1 / (1 + exp((V/mV + 56.9) / 4.9)) : 1
    # tau_m_A = (11.6 - (10.4 / (1 + exp((V/mV + 32.9) / -15.2)))) * ms : second
    # tau_h_A = (36.8 - (29.2 / (1 + exp((V/mV + 38.9) / -26.5)))) * ms : second

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
    P_target.Ca_target = Ca_target

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

    to_monitor = ['V', 'g_total', 'g_Na', 'I_Ca', 'g_K', 'Ca']
    traces = StateMonitor(P_target, to_monitor, record=True)

    net.add([spikes, traces])

    # ----------------------------------------------------
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
    I_Ca = np.asarray(traces.I_Ca_)
    calcium = np.asarray(traces.Ca)

    # and repack them
    results = {
        'ns': ns,
        'ts': ts,
        'times': times,
        'v_m': vm,
        'g_total': g_total,
        'calcium': calcium,
        'I_Ca': I_Ca,
        'g_Na': g_Na,
        'g_K': g_K
    }

    return results
