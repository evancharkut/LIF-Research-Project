"""Regenerate every figure in figures/ from the Python port.

Mirrors make_figures.m: each block below is one of the MATLAB demo scripts,
and prints the same summary numbers it does.

    python3 python/make_figures.py          # run from the repository root

Random seeds are fixed, but NumPy's generator is not MATLAB's, so the
noise-driven panels differ in detail from the MATLAB figures while agreeing
statistically.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lif import (filtered_noise, fI_theory, lif_params, lif_run, shift_trials,
                 spike_bits, spike_entropy, spike_events)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
DPI = 150


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved figures/{name}")


# --------------------------------------------------------------------------
# lif_dc.m -- response to a DC current
# --------------------------------------------------------------------------
def dc_step():
    dt = 0.05                       # [ms]
    duration = 1.0                  # [s]
    t_end = duration * 1000         # [ms]
    I_dc = 2.0                      # [nA]

    N = int(round(t_end / dt))
    I_e_vect = np.full(N, I_dc)
    V_vect, spike_times, t_vect = lif_run(I_e_vect, dt)

    rate = spike_times.size / duration
    print(f"I = {I_dc:.2f} nA  ->  {spike_times.size} spikes in "
          f"{duration:.2f} s  =  {rate:.1f} Hz")

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t_vect, I_e_vect, lw=1)
    ax[0].set_ylabel("Current (nA)")
    ax[0].set_ylim(0, I_dc * 1.5)
    ax[0].set_title(f"DC stimulus, I = {I_dc:.2f} nA  ({rate:.1f} Hz)")
    ax[1].plot(t_vect, V_vect, lw=0.7)
    ax[1].set_ylabel("Voltage (mV)")
    ax[2].vlines(spike_times, 0, 1, lw=0.8)
    ax[2].set_ylim(0, 1.5)
    ax[2].set_ylabel("Spikes")
    ax[2].set_xlabel("Time (ms)")
    ax[2].set_xlim(0, t_end)
    save(fig, "dc_step.png")


# --------------------------------------------------------------------------
# lif_fI_curve.m -- firing rate vs. DC amplitude, against the closed form
# --------------------------------------------------------------------------
def fI_curve():
    dt = 0.05                       # [ms]
    duration = 2.0                  # [s] longer run = smoother rate estimate
    t_end = duration * 1000
    N = int(round(t_end / dt))

    I_list = np.arange(0, 10.0001, 0.05)      # [nA]
    rates = np.zeros(I_list.size)
    for k, I in enumerate(I_list):
        _, spike_times, _ = lif_run(np.full(N, I), dt)
        rates[k] = spike_times.size / duration

    p = lif_params()
    theory = fI_theory(I_list, p)
    above = I_list > p.I_rheo

    print(f"Rheobase (minimum current to fire) = {p.I_rheo:.2f} nA")
    print(f"Max rate as I -> inf = {1000 / p.t_ref:.1f} Hz "
          f"(set by t_ref = {p.t_ref:g} ms)")
    print("Max |simulation - analytic| above rheobase = "
          f"{np.max(np.abs(rates[above] - theory[above])):.2f} Hz")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(I_list, rates, "o", ms=3, label="simulation")
    ax.plot(I_list, theory, "r-", lw=1.2, label="analytic")
    ax.axvline(p.I_rheo, color="k", ls="--", lw=1,
               label=f"rheobase = {p.I_rheo:g} nA")
    ax.set_xlabel("DC stimulus amplitude I (nA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Leaky integrate-and-fire: firing rate vs. DC amplitude")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "fI_curve.png")


# --------------------------------------------------------------------------
# The dt claim in README.md: the residual f-I error is threshold-crossing
# quantisation, not integration error, so it costs a fixed time per spike and
# therefore proportionally more at high rates.
# --------------------------------------------------------------------------
def dt_convergence():
    duration = 2.0                             # [s]
    dt_list = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    I_list = [2.0, 10.0]

    # The rate is measured from the mean inter-spike interval rather than from
    # the spike count. Counting spikes over a fixed window quantises the rate
    # to 1/duration = 0.5 Hz, which is larger than the effect being measured
    # and would swamp it at every dt.
    fig, ax = plt.subplots(figsize=(7, 5))
    for I in I_list:
        predicted = fI_theory([I])[0]
        err = []
        for dt in dt_list:
            N = int(round(duration * 1000 / dt))
            _, spike_times, _ = lif_run(np.full(N, I), dt)
            err.append(abs(1000.0 / np.diff(spike_times).mean() - predicted))
        print(f"I = {I:5.1f} nA (predicted {predicted:6.2f} Hz): error "
              + ", ".join(f"{e:.4f}" for e in err) + " Hz")
        ax.loglog(dt_list, err, "o-", label=f"I = {I:g} nA "
                                            f"({predicted:.0f} Hz)")

    # Detecting the crossing only at the end of a step biases every ISI upward
    # by up to one dt, which costs rate^2 * dt/1000 Hz. That fixed time error
    # is the slope-1 line the measured errors should track, and it is why the
    # same dt costs proportionally more at a high rate than at a low one.
    ref = np.array(dt_list)
    for I in I_list:
        r = fI_theory([I])[0]
        ax.loglog(ref, ref * r ** 2 / 1000, "k--", lw=1, alpha=0.6,
                  label=r"one $dt$ per spike" if I == I_list[0] else None)
    ax.set_xlabel("Time step dt (ms)")
    ax.set_ylabel("|simulated - analytic| rate (Hz)")
    ax.set_title("Residual f-I error is threshold-crossing quantisation")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    save(fig, "dt_convergence.png")


# --------------------------------------------------------------------------
# lif_filtered_noise.m -- frozen-noise raster
# --------------------------------------------------------------------------
def frozen_noise_raster():
    rng = np.random.default_rng(1)   # reproducible stimulus and trial noise

    dt = 0.05                        # [ms]
    duration = 1.0                   # [s]
    t_end = duration * 1000          # [ms]

    n_trials = 30
    rms_stim = 1.0                   # [nA] rms of the frozen component
    dc_offset = 1.5                  # [nA] sits at rheobase
    f_cut = 40.0                     # [Hz]
    rms_trial = 0.15                 # [nA] independent per-trial noise

    # Frozen stimulus: generated once, reused on every trial.
    I_frozen, t_vect = filtered_noise(t_end, dt, rms_stim, dc_offset, f_cut,
                                      rng=rng)

    spikes_by_trial = []
    for _ in range(n_trials):
        I_trial, _ = filtered_noise(t_end, dt, rms_trial, 0.0, f_cut, rng=rng)
        _, st, _ = lif_run(I_frozen + I_trial, dt)
        spikes_by_trial.append(st)

    n_spikes = np.array([st.size for st in spikes_by_trial])
    print(f"{n_trials} trials, {n_spikes.mean():.1f} +/- {n_spikes.std(ddof=1):.1f} "
          f"spikes per trial ({n_spikes.mean() / duration:.1f} Hz mean rate)")

    p = lif_params()
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    ax[0].plot(t_vect, I_frozen, lw=0.8)
    ax[0].axhline(p.I_rheo, color="k", ls="--", lw=1)
    ax[0].text(t_end * 0.995, p.I_rheo, " rheobase", va="bottom", ha="right",
               fontsize=8)
    ax[0].set_ylabel("Current (nA)")
    ax[0].set_title(f"Frozen filtered noise ({f_cut:g} Hz low-pass, "
                    f"{rms_stim:.1f} nA rms + {dc_offset:.1f} nA DC)")

    for k, st in enumerate(spikes_by_trial, start=1):
        if st.size:
            ax[1].vlines(st, k - 0.4, k + 0.4, color="k", lw=0.6)
    ax[1].set_xlim(0, t_end)
    ax[1].set_ylim(0.5, n_trials + 0.5)
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Trial")
    ax[1].set_title("Spike raster across trials")
    save(fig, "frozen_noise_raster.png")


# --------------------------------------------------------------------------
# lif_jitter_reliability.m -- timing precision vs. leak conductance
# --------------------------------------------------------------------------
def jitter_reliability():
    rng = np.random.default_rng(2)

    dt = 0.05                        # [ms]
    duration = 6.0                   # [s]
    t_end = duration * 1000          # [ms]
    n_trials = 10
    leak = np.arange(1, 9)           # normalised to the lif_params value

    sig_rms = 5.0                    # [nA] rms of the frozen signal
    sig_fcut = 100.0                 # [Hz]
    noise_rms = 1.0                  # [nA] rms of the per-trial noise
    noise_fcut = 20.0                # [Hz]
    dc = 0.0                         # [nA]

    opts = dict(binwidth=0.5, t_start=1000.0, peak_frac=0.10, rel_thresh=0.50)
    p0 = lif_params()

    rate = np.zeros(leak.size)
    jit = np.zeros(leak.size)
    rel = np.zeros(leak.size)

    for m, L in enumerate(leak):
        p = p0.with_leak(L)

        # Frozen stimulus: generated once per leak value, reused every trial.
        I_frozen, _ = filtered_noise(t_end, dt, sig_rms, dc, sig_fcut, rng=rng)

        spikes_by_trial = []
        for _ in range(n_trials):
            I_noise, _ = filtered_noise(t_end, dt, noise_rms, 0.0, noise_fcut,
                                        rng=rng)
            _, st, _ = lif_run(I_frozen + I_noise, dt, p)
            spikes_by_trial.append(st)

        ev = spike_events(spikes_by_trial, t_end, **opts)
        rate[m], jit[m], rel[m] = ev.rate, ev.jitter_mean, ev.reliability_mean

        print(f"leak {L:g} (tau {p.tau:4.2f} ms): {ev.rate:5.1f} Hz, "
              f"{ev.n_events:3d} events ({ev.n_multiple} discarded), "
              f"jitter {ev.jitter_mean:.3f} ms, "
              f"reliability {ev.reliability_mean:.2f}")

    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(leak, rate, "o-")
    ax[0].set_ylabel("Rate (Hz)")
    ax[0].set_title(f"Timing precision vs. leak ({n_trials} trials, "
                    f"{duration:g} s, {sig_rms:g} nA rms signal + "
                    f"{noise_rms:g} nA rms noise)")
    ax[1].plot(leak, jit, "o-")
    ax[1].set_ylabel("Jitter (ms)")
    ax[2].plot(leak, rel, "o-")
    ax[2].set_ylabel("Reliability")
    ax[2].set_xlabel("Leak (x default)")
    ax[2].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.3)
    save(fig, "jitter_reliability.png")


# --------------------------------------------------------------------------
# lif_information.m -- total entropy, noise entropy, information
# --------------------------------------------------------------------------
def information():
    rng = np.random.default_rng(5)

    dt = 0.1                        # [ms] converged; see the f-I validation
    t_bin = 1.0                     # [ms] one bin holds at most one spike here
    t_start = 1000.0                # [ms] discard the settling transient
    L_list = np.array([4, 5, 6, 8, 10, 12, 14, 16])      # word lengths [bins]
    L_fit = (6, 14)                 # range fitted for the 1/L -> 0 intercept

    dc, rms_stim, f_cut, rms_trial = 1.5, 1.0, 40.0, 0.15

    t_long = 200 * 1000.0           # [ms] single long trial, for total entropy
    t_trial = 16 * 1000.0           # [ms] frozen stimulus, repeated
    n_trials = 128                  # enough that the shift control is near zero

    # --- one long trial: total entropy ---
    # The per-trial noise is present here too: the total entropy has to be the
    # entropy of the responses the cell actually produces, noise included.
    I_long = (filtered_noise(t_long, dt, rms_stim, dc, f_cut, rng=rng)[0]
              + filtered_noise(t_long, dt, rms_trial, 0.0, f_cut, rng=rng)[0])
    spikes_long = lif_run(I_long, dt)[1]

    bits_long, merged_long = spike_bits(spikes_long, t_long, t_bin)
    bits_long = bits_long[:, int(t_start / t_bin):]
    S_long = spike_entropy(bits_long, t_bin, L_list)

    # --- many trials of a frozen stimulus: noise entropy ---
    I_frozen = filtered_noise(t_trial, dt, rms_stim, dc, f_cut, rng=rng)[0]
    spikes_by_trial = []
    for _ in range(n_trials):
        I_noise = filtered_noise(t_trial, dt, rms_trial, 0.0, f_cut, rng=rng)[0]
        spikes_by_trial.append(lif_run(I_frozen + I_noise, dt)[1])

    bits_tr, merged_tr = spike_bits(spikes_by_trial, t_trial, t_bin)
    bits_tr = bits_tr[:, int(t_start / t_bin):]
    S_tr = spike_entropy(bits_tr, t_bin, L_list)

    # The control: the same trials with the time-locking shifted away, where
    # the true information is zero. What comes back is the estimator's floor.
    S_ctrl = spike_entropy(shift_trials(bits_tr, rng), t_bin, L_list)
    floor = S_ctrl.info

    print(f"long run : {t_long/1000:g} s, 1 trial, {S_long.rate:.2f} Hz, "
          f"{merged_long} merged spikes")
    print(f"frozen   : {t_trial/1000:g} s x {n_trials} trials, "
          f"{S_tr.rate:.2f} Hz, {merged_tr} merged spikes")
    gap = 100 * abs(S_long.H_total[0] - S_tr.H_total[0]) / S_tr.H_total[0]
    print(f"total entropy from the two runs agrees to {gap:.1f}% at "
          f"L = {L_list[0]} ({S_long.H_total[0]:.1f} vs {S_tr.H_total[0]:.1f} bits/s)")

    print("\n  L   1/L   H_total  H_noise     info    floor   bits/spike  words seen")
    for m, L in enumerate(L_list):
        print(f"{L:3d}  {1/L:.3f}  {S_tr.H_total[m]:7.1f}  {S_tr.H_noise[m]:7.1f}  "
              f"{S_tr.info[m]:7.1f}  {floor[m]:7.1f}   "
              f"{S_tr.info[m]/S_tr.rate:8.2f}   {S_tr.n_seen[m]:6d}/{2**L}")

    # --- extrapolate each entropy rate to 1/L -> 0 ---
    x = 1.0 / L_list
    fit = (L_list >= L_fit[0]) & (L_list <= L_fit[1])
    c_t = np.polyfit(x[fit], S_tr.H_total[fit], 1)
    c_n = np.polyfit(x[fit], S_tr.H_noise[fit], 1)
    c_l = np.polyfit(x[fit], S_long.H_total[fit], 1)
    H_total_inf, H_noise_inf = c_t[1], c_n[1]
    info_inf = H_total_inf - H_noise_inf

    print(f"\nextrapolated to 1/L -> 0 over L = {L_fit[0]}..{L_fit[1]}:")
    print(f"  total entropy {H_total_inf:6.1f} bits/s  ({c_l[1]:.1f} from the long run)")
    print(f"  noise entropy {H_noise_inf:6.1f} bits/s")
    print(f"  information   {info_inf:6.1f} bits/s = "
          f"{info_inf/S_tr.rate:.2f} bits/spike at {S_tr.rate:.1f} Hz")

    # --- figure ---
    xf = np.array([0.0, x.max()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(x, S_tr.H_total, "o-", label="total (frozen trials)")
    ax[0].plot(x, S_long.H_total, "s--", label="total (one long trial)")
    ax[0].plot(x, S_tr.H_noise, "o-", label="noise")
    ax[0].plot(xf, np.polyval(c_t, xf), "k:")
    ax[0].plot(xf, np.polyval(c_n, xf), "k:")
    ax[0].plot([0, 0], [H_total_inf, H_noise_inf], "k*", ms=9)
    ax[0].set_xlabel("1 / word length")
    ax[0].set_ylabel("Entropy rate (bits/s)")
    ax[0].set_title(f"Entropy vs. word length ({n_trials} trials, {t_bin:g} ms bins)")
    ax[0].legend(loc="center right")

    ax[1].plot(x, S_tr.info, "o-", label="information")
    ax[1].plot(x, floor, "x-", label="shift control (floor)")
    ax[1].plot(xf, np.polyval(c_t - c_n, xf), "k:")
    ax[1].plot(0, info_inf, "k*", ms=9)
    ax[1].annotate(f"{info_inf:.0f} bits/s\n{info_inf/S_tr.rate:.2f} bits/spike",
                   (0, info_inf), textcoords="offset points", xytext=(10, -14))
    ax[1].set_xlabel("1 / word length")
    ax[1].set_ylabel("Information rate (bits/s)")
    ax[1].set_title("Total minus noise")
    ax[1].set_ylim(-0.06 * info_inf, 1.2 * info_inf)
    ax[1].legend(loc="center right")

    for a in ax:
        a.set_xlim(*xf)
        a.grid(alpha=0.3)
    save(fig, "information.png")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    for step in (dc_step, fI_curve, dt_convergence, frozen_noise_raster,
                 jitter_reliability, information):
        print(f"\n--- {step.__name__} ---")
        step()
