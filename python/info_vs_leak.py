"""Information, jitter and reliability vs. leak, averaged over many stimuli.

The leak sweep in make_figures.py uses one frozen stimulus per leak value and
10 trials, which is enough for jitter and reliability but not for the noise
entropy, and it confounds the leak with the stimulus: every leak value sees a
different input. Here both are fixed:

  - n_stim independent frozen stimuli, each presented for n_trials trials, so
    every point is an average over stimuli with a standard error across them;
  - every leak value sees the *same* n_stim stimuli and the *same* per-trial
    noise, so the difference between two leak values is the leak and nothing
    else (common random numbers).

The stimulus is the one from the jitter sweep -- 5 nA rms signal low-passed at
100 Hz, plus 1 nA rms per-trial noise low-passed at 20 Hz -- so jitter,
reliability and information all come from the same rasters. The leak is
modelled as in make_figures.py: R_m = R_0/leak, the neuromodulator changes the
cell and leaves the input alone.

For each stimulus and leak value the information is estimated as in
information(): total and noise entropy on 1 ms words, extrapolated to
1/L -> 0 over L = 6..14. The shift control is run on every raster too, so the
estimator's floor is reported alongside every estimate.

    python3 python/info_vs_leak.py            # run from the repository root
    python3 python/info_vs_leak.py --replot   # redraw from the saved .npz

Takes a few minutes on 8 cores. Writes figures/info_vs_leak.png, two
slide-sized figures (slide_leak_tradeoff.png, slide_leak_information.png) and
the per-stimulus numbers to figures/info_vs_leak.npz.
"""

import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lif import (filtered_noise, lif_params, lif_run, shift_trials, spike_bits,
                 spike_entropy, spike_events)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
DPI = 150

SEED = 7
n_stim = 10                      # independent frozen stimuli
n_trials = 100                   # trials of each
leak = np.arange(1, 9)           # normalised to the lif_params value

dt = 0.05                        # [ms]
t_trial = 11 * 1000.0            # [ms] 1 s transient + 10 s analysed
t_start = 1000.0                 # [ms]

sig_rms, sig_fcut = 5.0, 100.0   # [nA], [Hz] frozen signal
noise_rms, noise_fcut = 1.0, 20.0  # [nA], [Hz] per-trial noise
dc = 0.0                         # [nA]

t_bin = 1.0                      # [ms]
L_list = np.array([4, 5, 6, 8, 10, 12, 14, 16])
L_fit = (6, 14)
L_floor = 14                     # word length at which the floor is quoted

ev_opts = dict(binwidth=0.5, t_start=t_start, peak_frac=0.10, rel_thresh=0.50)

FIELDS = ("rate", "jitter", "reliability", "n_events", "H_total", "H_noise",
          "info", "info_L", "floor_L", "merged")
CURVES = ("H_total_L", "H_noise_L", "floor_curve")   # per word length


def extrapolate(S):
    """Total and noise entropy rates extrapolated to 1/L -> 0 [bits/s]."""
    x = 1.0 / L_list
    fit = (L_list >= L_fit[0]) & (L_list <= L_fit[1])
    H_total = np.polyfit(x[fit], S.H_total[fit], 1)[1]
    H_noise = np.polyfit(x[fit], S.H_noise[fit], 1)[1]
    return H_total, H_noise


def one_stimulus(seq):
    """Every leak value, driven by one frozen stimulus and one set of trials."""
    s_seq, n_seq, c_seq = seq.spawn(3)
    I_frozen = filtered_noise(t_trial, dt, sig_rms, dc, sig_fcut,
                              rng=np.random.default_rng(s_seq))[0]
    trial_seqs = n_seq.spawn(n_trials)
    p0 = lif_params()
    iL = int(np.flatnonzero(L_list == L_floor)[0])

    out = {f: np.zeros(leak.size) for f in FIELDS}
    out.update({f: np.zeros((leak.size, L_list.size)) for f in CURVES})
    for m, lk in enumerate(leak):
        p = p0.with_leak(lk)

        # The same noise on trial k at every leak value: regenerate it from
        # that trial's seed rather than hold 100 long vectors in memory.
        spikes_by_trial = []
        for ts in trial_seqs:
            I_noise = filtered_noise(t_trial, dt, noise_rms, 0.0, noise_fcut,
                                     rng=np.random.default_rng(ts))[0]
            spikes_by_trial.append(lif_run(I_frozen + I_noise, dt, p)[1])

        ev = spike_events(spikes_by_trial, t_trial, **ev_opts)

        bits, merged = spike_bits(spikes_by_trial, t_trial, t_bin)
        bits = bits[:, int(t_start / t_bin):]
        S = spike_entropy(bits, t_bin, L_list)
        S_ctrl = spike_entropy(shift_trials(bits, np.random.default_rng(c_seq)),
                               t_bin, L_list)
        H_total, H_noise = extrapolate(S)

        out["rate"][m] = S.rate
        out["jitter"][m] = ev.jitter_mean
        out["reliability"][m] = ev.reliability_mean
        out["n_events"][m] = ev.n_events
        out["H_total"][m] = H_total
        out["H_noise"][m] = H_noise
        out["info"][m] = H_total - H_noise
        out["info_L"][m] = S.info[iL]
        out["floor_L"][m] = S_ctrl.info[iL]
        out["merged"][m] = merged
        out["H_total_L"][m] = S.H_total
        out["H_noise_L"][m] = S.H_noise
        out["floor_curve"][m] = S_ctrl.info
    return out


def mean_sem(X):
    """Mean and standard error across stimuli (axis 0), ignoring NaNs."""
    n = np.sum(~np.isnan(X), axis=0)
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1) / np.sqrt(n)


def main():
    seqs = np.random.SeedSequence(SEED).spawn(n_stim)
    with Pool(min(n_stim, os.cpu_count())) as pool:
        per_stim = pool.map(one_stimulus, seqs)

    R = {f: np.array([o[f] for o in per_stim])       # stim x leak (x L)
         for f in FIELDS + CURVES}
    R["bits_per_spike"] = R["info"] / R["rate"]
    os.makedirs(FIGDIR, exist_ok=True)
    np.savez(os.path.join(FIGDIR, "info_vs_leak.npz"), leak=leak,
             L_list=L_list, **R)

    print(f"{n_stim} stimuli x {n_trials} trials x {t_trial/1000 - t_start/1000:g} s, "
          f"{t_bin:g} ms bins, fit L = {L_fit[0]}..{L_fit[1]}; mean +/- s.e.m.\n")
    print("leak  tau    rate (Hz)     jitter (ms)   reliability   "
          "info (bits/s)   bits/spike    floor@L14  merged")
    p0 = lif_params()
    for m, lk in enumerate(leak):
        cols = []
        for f, fmt in (("rate", "5.1f"), ("jitter", "5.3f"),
                       ("reliability", "5.3f"), ("info", "5.1f"),
                       ("bits_per_spike", "5.2f")):
            mu, se = mean_sem(R[f][:, m:m + 1])
            cols.append(f"{mu[0]:{fmt}} +/- {se[0]:{fmt}}")
        print(f"{lk:3d}  {p0.with_leak(lk).tau:5.2f}  " + "  ".join(cols)
              + f"   {np.mean(R['floor_L'][:, m]):5.1f}   "
              f"{int(R['merged'][:, m].sum())}")

    plot(R)


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), dpi=DPI)
    plt.close(fig)
    print(f"saved figures/{name}")


def panel(a, R, f, label):
    """One thin grey line per stimulus, mean +/- s.e.m. on top."""
    a.plot(leak, R[f].T, color="0.75", lw=0.8)
    mu, se = mean_sem(R[f])
    a.errorbar(leak, mu, yerr=se, fmt="o-", color="C0", lw=2, ms=6,
               capsize=3, zorder=3)
    a.set_ylabel(label)
    a.set_xlabel("Leak (x default)")
    a.set_xticks(leak)
    a.grid(alpha=0.3)
    return mu


def plot(R):
    # --- summary figure: every quantity side by side ---
    panels = (("rate", "Rate (Hz)"),
              ("jitter", "Jitter (ms)"),
              ("reliability", "Reliability"),
              ("info", "Information (bits/s)"),
              ("bits_per_spike", "Information (bits/spike)"))
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), sharex=True)
    for a, (f, label) in zip(ax, panels):
        panel(a, R, f, label)
    ax[2].set_ylim(0, 1)
    for a in ax[3:]:
        a.set_ylim(bottom=0)
    fig.suptitle(f"Leak sweep, averaged over {n_stim} frozen stimuli x "
                 f"{n_trials} trials (grey: each stimulus; blue: mean +/- s.e.m.)")
    fig.tight_layout()
    save(fig, "info_vs_leak.png")

    # --- the same results as two slide-sized figures, in large type ---
    with plt.rc_context({"font.size": 15, "axes.titlesize": 16}):
        # the trade-off: sharper timing, fewer spikes
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        panel(ax[0], R, "jitter", "Jitter (ms)")
        ax[0].set_title("Spike timing gets sharper")
        ax[0].set_ylim(bottom=0)
        panel(ax[1], R, "rate", "Firing rate (Hz)")
        ax[1].set_title("...but the cell fires less")
        ax[1].set_ylim(bottom=0)
        fig.tight_layout()
        save(fig, "slide_leak_tradeoff.png")

        # the payoff: information per second peaks, per spike keeps rising
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        mu = panel(ax[0], R, "info", "Information (bits/s)")
        k = int(np.argmax(mu))
        ax[0].annotate(f"peak: {mu[k]:.0f} bits/s\nat {leak[k]:g}x leak",
                       (leak[k], mu[k]), xytext=(leak[k] + 1.5, 1.14 * mu[k]),
                       va="center", arrowprops=dict(arrowstyle="->", color="0.3"))
        ax[0].set_title("Information per second")
        ax[0].set_ylim(0, 1.3 * mu.max())
        panel(ax[1], R, "bits_per_spike", "Information (bits/spike)")
        ax[1].set_title("Information per spike")
        ax[1].set_ylim(bottom=0)
        fig.tight_layout()
        save(fig, "slide_leak_information.png")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        # Redraw from the saved results without rerunning the simulations.
        d = np.load(os.path.join(FIGDIR, "info_vs_leak.npz"))
        plot({f: d[f] for f in d.files})
    else:
        main()
