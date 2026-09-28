"""Frozen-noise raster for a slide, on the stimulus the rest of the talk uses.

frozen_noise_raster.png in make_figures.py is drawn from a different stimulus
(1 nA rms around rheobase, 30 trials) and starts at t = 0, inside the filter's
start-up transient. This draws the default cell (leak = 1) on one of the
info_vs_leak.py stimuli instead -- 5 nA rms signal low-passed at 100 Hz, plus
1 nA rms per-trial noise low-passed at 20 Hz, 100 trials -- from the same seeds,
so these are the rasters the jitter, reliability and information come from.

The half-second shown holds the stretch jitter_vs_trials.py zooms into.

    python3 python/raster_slide.py    # run from the repository root

Takes about 10 s. Writes figures/slide_frozen_noise_raster.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import info_vs_leak as ivl
import jitter_vs_trials as jvt
from lif import filtered_noise, lif_params, lif_run

WINDOW = (6000.0, 6500.0)         # [ms] well past the 1 s transient


def main():
    seq = np.random.SeedSequence(ivl.SEED).spawn(ivl.n_stim)[jvt.EX_STIM]
    s_seq, n_seq, _ = seq.spawn(3)
    I_frozen, t_vect = filtered_noise(ivl.t_trial, ivl.dt, ivl.sig_rms, ivl.dc,
                                      ivl.sig_fcut,
                                      rng=np.random.default_rng(s_seq))
    p = lif_params()
    spikes_by_trial = []
    for ts in n_seq.spawn(ivl.n_trials):
        I_noise = filtered_noise(ivl.t_trial, ivl.dt, ivl.noise_rms, 0.0,
                                 ivl.noise_fcut, rng=np.random.default_rng(ts))[0]
        spikes_by_trial.append(lif_run(I_frozen + I_noise, ivl.dt, p)[1])

    lo, hi = WINDOW
    n_spk = np.array([np.sum((st >= lo) & (st < hi)) for st in spikes_by_trial])
    print(f"stimulus {jvt.EX_STIM}, {ivl.n_trials} trials, {lo/1000:g}-{hi/1000:g} s: "
          f"{n_spk.mean():.1f} +/- {n_spk.std(ddof=1):.1f} spikes per trial "
          f"({n_spk.mean() / ((hi - lo) / 1000):.1f} Hz)")

    in_win = (t_vect >= lo) & (t_vect <= hi)
    with plt.rc_context({"font.size": 15, "axes.titlesize": 16}):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.2]})
        ax[0].plot(t_vect[in_win] - lo, I_frozen[in_win], color="C0", lw=1.4)
        ax[0].axhline(p.I_rheo, color="k", ls="--", lw=1)
        ax[0].text(hi - lo + 4, p.I_rheo, "rheobase", va="center", ha="left",
                   fontsize=12, clip_on=False)
        ax[0].set_ylabel("Current (nA)")
        ax[0].set_title(f"Frozen stimulus, the same on every trial "
                        f"({ivl.sig_rms:g} nA rms)", loc="left")

        for k, st in enumerate(spikes_by_trial, start=1):
            st = st[(st >= lo) & (st <= hi)] - lo
            ax[1].vlines(st, k - 0.5, k + 0.5, color="k", lw=1.0)
        ax[1].set_xlim(0, hi - lo)
        ax[1].set_ylim(0.5, ivl.n_trials + 0.5)
        ax[1].set_yticks([1, 25, 50, 75, 100])
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylabel("Trial")
        ax[1].set_title(f"Spikes on {ivl.n_trials} trials, each with its own "
                        f"{ivl.noise_rms:g} nA rms noise added", loc="left")
        for a in ax:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(ivl.FIGDIR, "slide_frozen_noise_raster.png"),
                    dpi=ivl.DPI)
        plt.close(fig)
    print("saved figures/slide_frozen_noise_raster.png")


if __name__ == "__main__":
    main()
