"""How many trials the jitter measure needs.

Jitter and reliability come from events found in the PSTH pooled over trials
(spike_events). With few trials that PSTH is sparse: a broad event has empty
bins inside it, so it is split into several narrow pieces, most of them below
the 50% reliability cut. What survives the cut is the tight core, and the mean
jitter comes out too low.

This takes the leak = 1 rasters of info_vs_leak.py -- the same 10 stimuli and
the same per-trial noise, so the 100-trial numbers are that sweep's -- and
measures jitter and reliability from n of the 100 trials, averaged over the
100 // n disjoint groups of n trials in each stimulus.

    python3 python/jitter_vs_trials.py            # run from the repository root
    python3 python/jitter_vs_trials.py --replot   # redraw from the saved .npz

Takes about 15 s. Writes figures/slide_jitter_vs_trials.png and the numbers to
figures/jitter_vs_trials.npz.
"""

import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import info_vs_leak as ivl
from lif import filtered_noise, lif_params, lif_run, spike_events

FIGDIR = ivl.FIGDIR
N_LIST = np.array([3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 70, 100])
N_FEW = 10                        # the trial count the old sweep used

# The stretch of one stimulus drawn as the example; its edges sit in quiet
# gaps, so no event is cut by them.
EX_STIM = 1
EX_WINDOW = (6273.0, 6299.0)      # [ms]
EX_MARGIN = 5.0                   # [ms] spikes kept either side of the window


def one_stimulus(seq):
    """100 trials of one stimulus at the default leak, drawn from the same
    seeds as info_vs_leak.one_stimulus, then measured from n of them."""
    s_seq, n_seq, _ = seq.spawn(3)
    I_frozen = filtered_noise(ivl.t_trial, ivl.dt, ivl.sig_rms, ivl.dc,
                              ivl.sig_fcut, rng=np.random.default_rng(s_seq))[0]
    p = lif_params().with_leak(1)
    spikes_by_trial = []
    for ts in n_seq.spawn(ivl.n_trials):
        I_noise = filtered_noise(ivl.t_trial, ivl.dt, ivl.noise_rms, 0.0,
                                 ivl.noise_fcut, rng=np.random.default_rng(ts))[0]
        spikes_by_trial.append(lif_run(I_frozen + I_noise, ivl.dt, p)[1])

    jit, rel, n_ev = (np.zeros(N_LIST.size) for _ in range(3))
    for i, n in enumerate(N_LIST):
        ev = [spike_events(spikes_by_trial[g * n:(g + 1) * n], ivl.t_trial,
                           **ivl.ev_opts)
              for g in range(ivl.n_trials // n)]
        jit[i] = np.nanmean([e.jitter_mean for e in ev])
        rel[i] = np.nanmean([e.reliability_mean for e in ev])
        n_ev[i] = np.mean([e.n_events for e in ev])
    return jit, rel, n_ev, spikes_by_trial


def main():
    seqs = np.random.SeedSequence(ivl.SEED).spawn(ivl.n_stim)
    with Pool(min(ivl.n_stim, os.cpu_count())) as pool:
        out = pool.map(one_stimulus, seqs)

    R = {f: np.array([o[k] for o in out])                # stim x n
         for k, f in enumerate(("jitter", "reliability", "n_events"))}
    lo, hi = EX_WINDOW[0] - EX_MARGIN, EX_WINDOW[1] + EX_MARGIN
    ex = [st[(st > lo) & (st < hi)] for st in out[EX_STIM][3]]
    R["ex_spikes"] = np.concatenate(ex)
    R["ex_counts"] = np.array([e.size for e in ex])
    R["N_LIST"] = N_LIST
    np.savez(os.path.join(FIGDIR, "jitter_vs_trials.npz"), **R)

    print(f"leak = 1, {ivl.n_stim} stimuli; each n averages the "
          f"{ivl.n_trials} // n disjoint groups of n trials; mean +/- s.e.m.\n")
    print("trials   jitter (ms)       reliability       events")
    stats = [ivl.mean_sem(R[f]) for f in ("jitter", "reliability")]
    for i, n in enumerate(N_LIST):
        cols = [f"{mu[i]:5.3f} +/- {se[i]:5.3f}" for mu, se in stats]
        print(f"{n:5d}    " + "   ".join(cols)
              + f"   {R['n_events'][:, i].mean():6.1f}")

    # The 100-trial column must be info_vs_leak's leak = 1 column exactly.
    saved = os.path.join(FIGDIR, "info_vs_leak.npz")
    if os.path.exists(saved):
        same = np.allclose(R["jitter"][:, -1], np.load(saved)["jitter"][:, 0])
        print(f"\n100-trial jitter matches info_vs_leak.npz: {same}")

    plot(R)


def event_spans(spikes_by_trial, n_trials):
    """Events of the example stretch, with the PSTH run each one covers.

    spike_events returns event centres, not extents. An event is the run of
    contiguous non-empty bins around a seed bin, so its extent is the run that
    holds its centre, on the same 0.5 ms grid starting at 0.
    """
    lo, hi = EX_WINDOW[0] - EX_MARGIN, EX_WINDOW[1] + EX_MARGIN
    opts = dict(ivl.ev_opts, t_start=lo)
    ev = spike_events(spikes_by_trial, hi, **opts)
    bw = opts["binwidth"]
    edges = np.arange(0.0, hi + bw / 2, bw)
    X, _ = np.histogram(np.concatenate(spikes_by_trial), bins=edges)
    spans = []
    for t, j, r in zip(ev.times, ev.jitter, ev.reliability):
        k = int(t // bw)
        a, b = k, k
        while a > 0 and X[a - 1] > 0:
            a -= 1
        while b < X.size - 1 and X[b + 1] > 0:
            b += 1
        if edges[a] >= EX_WINDOW[0] and edges[b + 1] <= EX_WINDOW[1]:
            spans.append((edges[a], edges[b + 1], j, r))
    return spans, edges, X


def raster(a, spikes_by_trial, title):
    """Spike ticks, events shaded: blue if counted, grey if below the cut."""
    n = len(spikes_by_trial)
    spans, _, _ = event_spans(spikes_by_trial, n)
    for x0, x1, j, r in spans:
        counted = r > ivl.ev_opts["rel_thresh"]
        a.axvspan(x0 - EX_WINDOW[0], x1 - EX_WINDOW[0], lw=0,
                  color="C0" if counted else "0.6",
                  alpha=0.22 if counted else 0.25)
        if counted:
            a.text((x0 + x1) / 2 - EX_WINDOW[0], n + 0.5 + 0.04 * n,
                   f"{j:.2f} ms", ha="center", va="bottom", fontsize=12,
                   color="C0")
    for k, st in enumerate(spikes_by_trial, start=1):
        st = st[(st >= EX_WINDOW[0]) & (st <= EX_WINDOW[1])] - EX_WINDOW[0]
        h = 0.4 if n <= 20 else 0.5
        a.vlines(st, k - h, k + h, color="k", lw=1.2 if n <= 20 else 1.8)
    a.set_ylim(0.5, n + 0.5 + 0.2 * n)
    a.set_xlim(0, EX_WINDOW[1] - EX_WINDOW[0])
    a.set_yticks([1, n])
    a.set_ylabel("Trial")
    a.set_title(title, loc="left")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def plot(R):
    c = np.concatenate(([0], np.cumsum(R["ex_counts"])))
    ex = [R["ex_spikes"][c[k]:c[k + 1]] for k in range(c.size - 1)]
    n_list = R["N_LIST"]

    with plt.rc_context({"font.size": 15, "axes.titlesize": 16}):
        fig = plt.figure(figsize=(15, 5.8))
        gs = fig.add_gridspec(2, 2, width_ratios=(1.1, 1),
                              height_ratios=(1, 1.8), hspace=0.5, wspace=0.22)

        # --- the example: one stretch, 10 trials vs. all 100 ---
        a10 = fig.add_subplot(gs[0, 0])
        a100 = fig.add_subplot(gs[1, 0], sharex=a10)
        raster(a10, ex[:N_FEW], f"{N_FEW} trials: the broad event breaks up")
        raster(a100, ex, f"{len(ex)} trials: a tight event and a broad one")
        a100.set_xlabel("Time (ms)")
        a10.tick_params(labelbottom=False)
        a100.plot([], [], "s", color="C0", alpha=0.4, ms=12,
                  label="event (counted)")
        a100.plot([], [], "s", color="0.6", alpha=0.45, ms=12,
                  label="piece < 50% reliable (dropped)")
        a10.legend(*a100.get_legend_handles_labels(), loc="lower left",
                   bbox_to_anchor=(0.0, 1.25), ncol=2, frameon=False,
                   fontsize=12, handletextpad=0.2, columnspacing=1.0)

        # --- the fix: jitter vs. number of trials ---
        a = fig.add_subplot(gs[:, 1])
        a.plot(n_list, R["jitter"].T, color="0.75", lw=0.8)
        mu, se = ivl.mean_sem(R["jitter"])
        a.errorbar(n_list, mu, yerr=se, fmt="o-", color="C0", lw=2, ms=6,
                   capsize=3, zorder=3)
        a.set_xscale("log")
        a.set_xticks([3, 5, 10, 20, 30, 50, 100])
        a.set_xticklabels([3, 5, 10, 20, 30, 50, 100])
        a.minorticks_off()
        a.set_ylim(0, 1.2)
        a.set_xlabel("Trials per stimulus")
        a.set_ylabel("Jitter (ms)")
        a.grid(alpha=0.3)

        i10 = int(np.flatnonzero(n_list == N_FEW)[0])
        a.axhline(mu[-1], color="0.3", ls="--", lw=1)
        a.annotate(f"{N_FEW} trials: {mu[i10]:.2f} ms", (N_FEW, mu[i10]),
                   xytext=(N_FEW * 1.5, mu[i10] - 0.28), va="center",
                   arrowprops=dict(arrowstyle="->", color="0.3"))
        a.annotate(f"{n_list[-1]} trials (used): {mu[-1]:.2f} ms",
                   (n_list[-1], mu[-1]), xytext=(n_list[-1], mu[-1] + 0.08),
                   ha="right", va="center")
        a.axvspan(30, n_list[-1] * 1.15, color="C2", alpha=0.08, lw=0)
        a.set_xlim(n_list[0] / 1.15, n_list[-1] * 1.15)
        a.set_title("Jitter settles by ~30 trials")

        save = os.path.join(FIGDIR, "slide_jitter_vs_trials.png")
        fig.savefig(save, dpi=ivl.DPI, bbox_inches="tight")
        plt.close(fig)
        print("saved figures/slide_jitter_vs_trials.png")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        # Redraw from the saved results without rerunning the simulations.
        d = np.load(os.path.join(FIGDIR, "jitter_vs_trials.npz"))
        plot({f: d[f] for f in d.files})
    else:
        main()
