"""Leaky integrate-and-fire neuron: model, stimulus generator, event analysis.

A direct port of the MATLAB implementation in the repository root. Function
names, argument order and defaults follow the .m files one for one, so the two
can be compared line by line:

    lif_params.m     -> lif_params()
    lif_run.m        -> lif_run()
    filtered_noise.m -> filtered_noise()
    spike_events.m   -> spike_events()

Units are self-consistent: R_m in Mohm and I in nA give I*R_m in mV, and
tau = R_m*C_m with C_m in nF gives tau in ms.

The one thing the port cannot reproduce is MATLAB's random number stream, so
noise-driven results agree statistically but not sample for sample.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, filtfilt


# --------------------------------------------------------------------------
# lif_params.m
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Default parameters for the leaky integrate-and-fire neuron.

    Single source of truth for the model constants, so that lif_run and the
    analytic comparisons in the f-I curve cannot drift apart.
    """

    E_rest: float = -65.0     # resting potential                    [mV]
    V_thresh: float = -50.0   # spike threshold                      [mV]
    V_reset: float = -70.0    # post-spike reset                     [mV]
    V_spike: float = 20.0     # drawn spike height, cosmetic only    [mV]
    R_m: float = 10.0         # membrane resistance                  [Mohm]
    C_m: float = 1.0          # membrane capacitance                 [nF]
    V_0: float = -70.0        # initial condition                    [mV]
    t_ref: float = 3.0        # absolute refractory period           [ms]

    @property
    def tau(self):
        """Membrane time constant [ms]."""
        return self.R_m * self.C_m

    @property
    def I_rheo(self):
        """Rheobase: minimum DC current that makes the cell fire [nA]."""
        return (self.V_thresh - self.E_rest) / self.R_m

    def with_leak(self, leak):
        """The same cell with the leak conductance scaled: R_m = R_0/leak."""
        return replace(self, R_m=self.R_m / leak)


def lif_params():
    return Params()


# --------------------------------------------------------------------------
# lif_run.m
# --------------------------------------------------------------------------
def lif_run(I_e_vect, dt, p=None):
    """Leaky integrate-and-fire neuron driven by an arbitrary current.

    I_e_vect  injected current, one value per time step   [nA]
    dt        time step                                   [ms]
    p         Params; defaults to lif_params()

    Returns (V_vect, spike_times, t_vect). Spikes are drawn at p.V_spike in
    V_vect for display only; the integration state is a separate scalar, so
    the drawn spike never feeds back into the dynamics.

    The update is exponential Euler, which is exact for a current held
    constant across the step, so the result is insensitive to dt well before
    forward Euler would have converged.
    """
    if p is None:
        p = lif_params()

    I_e_vect = np.asarray(I_e_vect, dtype=float)
    N = I_e_vect.size
    t_vect = np.arange(1, N + 1) * dt

    # Precompute the per-step steady state and the decay factor; the loop
    # itself has to stay sequential because of the threshold test. Both are
    # held as Python lists/floats, which the loop indexes far faster than it
    # can index a numpy array.
    V_inf_list = (I_e_vect * p.R_m + p.E_rest).tolist()
    decay = float(np.exp(-dt / p.tau))
    n_ref = int(round(p.t_ref / dt))
    V_thresh, V_reset, V_spike = p.V_thresh, p.V_reset, p.V_spike

    V_out = [0.0] * N
    V_out[0] = p.V_0
    V = float(p.V_0)

    spike_idx = []
    ref_count = 0

    for i in range(1, N):
        if ref_count > 0:
            # --- absolute refractory period: clamp at reset, ignore input ---
            V = V_reset
            V_out[i] = V
            ref_count -= 1
        else:
            # --- exponential Euler step toward I*R + E_rest ---
            V_inf = V_inf_list[i]
            V = V_inf + (V - V_inf) * decay

            if V > V_thresh:
                V_out[i] = V_spike         # draw the spike at THIS index
                spike_idx.append(i)
                V = V_reset
                ref_count = n_ref
            else:
                V_out[i] = V

    V_vect = np.array(V_out)
    spike_times = t_vect[np.array(spike_idx, dtype=int)] if spike_idx else np.array([])
    return V_vect, spike_times, t_vect


def fI_theory(I_list, p=None):
    """Analytic firing rate for a DC current [Hz]; 0 below rheobase.

    Between spikes V relaxes toward I*R_m + E_rest starting from V_reset, so
        ISI = t_ref + tau*ln[(I*R+E_rest-V_reset)/(I*R+E_rest-V_thresh)]
    """
    if p is None:
        p = lif_params()
    I_list = np.asarray(I_list, dtype=float)
    V_inf = I_list * p.R_m + p.E_rest
    rate = np.zeros_like(V_inf)
    fire = V_inf > p.V_thresh
    ISI = p.tau * np.log(
        (V_inf[fire] - p.V_reset) / (V_inf[fire] - p.V_thresh)
    )
    rate[fire] = 1000.0 / (ISI + p.t_ref)
    return rate


# --------------------------------------------------------------------------
# filtered_noise.m
# --------------------------------------------------------------------------
def filtered_noise(t_end, dt, rms_target=1.0, dc_offset=1.5, f_cut=40.0,
                   order=4, rng=None):
    """Band-limited Gaussian noise current for driving the LIF neuron.

    t_end       duration                                   [ms]
    dt          time step                                  [ms]
    rms_target  rms of the fluctuating component           [nA]
    dc_offset   constant offset                            [nA]
    f_cut       low-pass cutoff                            [Hz]
    order       Butterworth order

    The filter is applied with filtfilt, so the stimulus is zero-phase and the
    effective attenuation is that of a filter of twice the stated order. The
    fluctuating component is scaled *after* filtering, so rms_target is the rms
    of the delivered stimulus, not of the white noise going in.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_vect = np.arange(1, int(round(t_end / dt)) + 1) * dt
    fs = 1000.0 / dt                      # sampling rate [Hz]  (dt is in ms)

    if f_cut >= fs / 2:
        raise ValueError(
            f"Cutoff ({f_cut} Hz) must be below Nyquist ({fs / 2} Hz)."
        )

    I_white = rng.standard_normal(t_vect.size)

    b, a = butter(order, f_cut / (fs / 2), btype="low")
    # padlen chosen to match MATLAB's filtfilt, which uses 3*(nfilt-1).
    I_filt = filtfilt(b, a, I_white, padlen=3 * (max(len(a), len(b)) - 1))

    # Scale the fluctuating part to the requested rms, then add the DC offset.
    I_filt = I_filt - I_filt.mean()
    I_filt = I_filt * (rms_target / I_filt.std(ddof=1))
    return I_filt + dc_offset, t_vect


# --------------------------------------------------------------------------
# spike_events.m
# --------------------------------------------------------------------------
@dataclass
class Events:
    times: np.ndarray            # event centre, mean spike time within it [ms]
    jitter: np.ndarray           # s.d. of the spike times within the event [ms]
    reliability: np.ndarray      # fraction of trials that contributed a spike
    jitter_mean: float           # mean jitter over events above rel_thresh
    reliability_mean: float      # mean reliability over those events
    n_multiple: int              # events discarded: some trial fired twice
    n_events: int
    rate: float                  # pooled firing rate over the counted window [Hz]


def spike_events(spikes_by_trial, t_end, binwidth=0.5, t_start=1000.0,
                 peak_frac=0.10, rel_thresh=0.50):
    """Group spikes across trials into events; jitter and reliability.

    Pools the spikes from every trial into one peri-stimulus time histogram,
    finds the repeatable "events" in it, and measures how precisely and how
    dependably each event is fired.

    spikes_by_trial  list of spike-time vectors, one per trial     [ms]
    t_end            trial duration                                [ms]
    binwidth         PSTH bin width                                [ms]
    t_start          ignore spikes before this, to skip the settling transient
    peak_frac        a bin holding at least this fraction of the trials seeds
                     an event
    rel_thresh       events below this reliability are left out of the summary
                     means, but are still returned per-event

    The event-finding rule is taken from Billimoria et al. (2006): a bin busy
    enough to be a peak is grown outwards over contiguous non-empty bins, and
    the resulting run of bins is one event. An event in which any single trial
    fired more than once is thrown away rather than counted, since its spread
    would measure the interval between two spikes rather than the trial-to-
    trial jitter of one.
    """
    if t_start >= t_end:
        raise ValueError(
            f"t_start ({t_start} ms) is not before t_end ({t_end} ms), so "
            "every spike would be discarded."
        )

    n_trials = len(spikes_by_trial)

    # --- pool the trials, remembering which trial each spike came from ---
    # The trial index is what makes the "fired twice in one event" test exact;
    # the original matched spike times against a padded array with INTERSECT,
    # which relies on floating-point times comparing equal.
    pooled, trial_id = [], []
    for k, st in enumerate(spikes_by_trial, start=1):
        st = np.asarray(st, dtype=float).ravel()
        st = st[st > t_start]
        pooled.append(st)
        trial_id.append(np.full(st.size, k))
    pooled = np.concatenate(pooled) if pooled else np.array([])
    trial_id = np.concatenate(trial_id) if trial_id else np.array([])

    rate = 1000.0 * pooled.size / ((t_end - t_start) * n_trials)
    empty = Events(np.array([]), np.array([]), np.array([]),
                   np.nan, np.nan, 0, 0, rate)
    if pooled.size == 0:
        return empty

    # --- peri-stimulus time histogram ---
    edges = np.arange(0.0, t_end + binwidth / 2, binwidth)
    if edges[-1] < t_end:
        edges = np.append(edges, edges[-1] + binwidth)
    n_bins = edges.size - 1
    X, _ = np.histogram(pooled, bins=edges)
    # 1-based bin index, matching MATLAB's DISCRETIZE.
    bin_of = np.clip(np.digitize(pooled, edges), 1, n_bins)

    # --- grow each peak bin into an event ---
    peak_bins = np.flatnonzero(X >= peak_frac * n_trials) + 1   # 1-based

    ev_time, ev_jit, ev_rel = [], [], []
    n_multiple = 0
    last_hi = 0                       # last bin already absorbed into an event

    for pb in peak_bins:
        if pb <= last_hi:
            continue                  # already inside the previous event

        lo = pb
        while lo > 1 and X[lo - 2] > 0:
            lo -= 1
        hi = pb
        while hi < n_bins and X[hi] > 0:
            hi += 1
        last_hi = hi

        in_ev = (bin_of >= lo) & (bin_of <= hi)
        times = pooled[in_ev]
        trials = trial_id[in_ev]

        # Discard the event if any trial put more than one spike in it.
        counts = np.bincount(trials.astype(int), minlength=n_trials + 1)[1:]
        if counts.max() > 1:
            n_multiple += 1
            continue

        ev_time.append(times.mean())
        # MATLAB's STD normalises by N-1 and returns 0 for a single sample.
        ev_jit.append(times.std(ddof=1) if times.size > 1 else 0.0)
        ev_rel.append(times.size / n_trials)

    ev_time = np.array(ev_time)
    ev_jit = np.array(ev_jit)
    ev_rel = np.array(ev_rel)

    good = ev_rel > rel_thresh
    jitter_mean = ev_jit[good].mean() if good.any() else np.nan
    reliability_mean = ev_rel[good].mean() if good.any() else np.nan

    return Events(ev_time, ev_jit, ev_rel, jitter_mean, reliability_mean,
                  n_multiple, ev_time.size, rate)


# --------------------------------------------------------------------------
# spike_bits.m
# --------------------------------------------------------------------------
def spike_bits(spikes_by_trial, t_end, t_bin=1.0):
    """Binarise spike trains into a trials x bins 0/1 matrix.

    spikes_by_trial  list of spike-time vectors, one per trial, or a single
                     vector for a one-trial run                      [ms]
    t_end            trial duration                                  [ms]
    t_bin            bin width                                       [ms]

    Returns (bits, n_merged). Bin k covers ((k-1)*t_bin, k*t_bin], matching
    ceil(t/t_bin), and a trailing partial bin is dropped.

    n_merged counts spikes that landed in an already-occupied bin. The direct
    method assumes a bin is small enough to hold at most one spike; with an
    absolute refractory period of t_ref the safe choice is t_bin <= t_ref, and
    n_merged > 0 says that choice has been broken and the words no longer
    carry the full spike count.
    """
    if isinstance(spikes_by_trial, np.ndarray) or not isinstance(
            spikes_by_trial, (list, tuple)):
        spikes_by_trial = [spikes_by_trial]

    n_trials = len(spikes_by_trial)
    n_bins = int(np.floor(t_end / t_bin))
    if n_bins < 1:
        raise ValueError(
            f"t_bin ({t_bin} ms) is larger than t_end ({t_end} ms)."
        )

    bits = np.zeros((n_trials, n_bins), dtype=np.uint8)
    n_merged = 0
    for k, st in enumerate(spikes_by_trial):
        st = np.asarray(st, dtype=float).ravel()
        st = st[(st > 0) & (st <= n_bins * t_bin)]
        idx = np.maximum(1, np.ceil(st / t_bin).astype(int)) - 1
        n_merged += idx.size - np.unique(idx).size
        bits[k, idx] = 1
    return bits, n_merged


# --------------------------------------------------------------------------
# spike_entropy.m
# --------------------------------------------------------------------------
@dataclass
class Entropy:
    L: np.ndarray                # word length                        [bins]
    T: np.ndarray                # word duration                      [s]
    rate: float                  # mean firing rate                   [Hz]
    n_trials: int
    t_bin: float
    H_total: np.ndarray          # total entropy rate                 [bits/s]
    H_noise: np.ndarray          # noise entropy rate                 [bits/s]
    info: np.ndarray             # H_total - H_noise                  [bits/s]
    H_total_naive: np.ndarray    # before the finite-sample correction
    H_noise_naive: np.ndarray
    n_words: np.ndarray          # samples behind each total-entropy estimate
    n_seen: np.ndarray           # distinct words actually observed

    @property
    def inv_L(self):
        return 1.0 / self.L

    def per_spike(self, field):
        """A rate in bits/s expressed in bits per spike."""
        return getattr(self, field) / self.rate


def _ent_cols(X):
    """Plug-in entropy of each column of X, in bits.

    Each of the c members of a group of identical entries contributes
    -(1/n)*log2(c/n), so the group as a whole contributes -(c/n)*log2(c/n) --
    which is the plug-in sum -sum p*log2(p). Summing over elements instead of
    over groups is what lets this be done without a Python loop.
    """
    X = X.reshape(X.shape[0], -1)
    n, m = X.shape
    if n < 2:
        return np.zeros(m)
    Z = np.sort(X, axis=0)
    new = np.empty(Z.shape, dtype=bool)
    new[0] = True
    new[1:] = Z[1:] != Z[:-1]
    g = np.cumsum(new, axis=0) - 1                  # 0-based group index
    lin = g + np.arange(m) * n                      # into an n x m count grid
    counts = np.bincount(lin.ravel(), minlength=n * m)
    c = counts[lin]                                 # group size of each element
    return -np.sum(np.log2(c / n), axis=0) / n


def _word_ids(bits, L, overlap=True):
    """Each length-L binary word as one integer, trials x start positions."""
    n_bins = bits.shape[1]
    n_pos = n_bins - L + 1
    if n_pos < 1:
        raise ValueError(f"word length {L} exceeds the {n_bins} bins available")
    ids = np.zeros((bits.shape[0], n_pos), dtype=np.int64)
    for k in range(L):
        ids = ids * 2 + bits[:, k:k + n_pos]
    return ids if overlap else ids[:, ::L]


def _extrap(H_list, N_list, H_full):
    """Intercept of a straight line through entropy vs. 1/sample-count."""
    if len(H_list) < 2:
        return H_full
    return float(np.polyfit(1.0 / np.asarray(N_list, float),
                            np.asarray(H_list, float), 1)[1])


def _extrap_samples(v, n_fractions):
    """Total entropy extrapolated to an infinite number of words."""
    N = v.size
    H_list, N_list = [], []
    for j in range(n_fractions):
        m = 2 ** j                                  # disjoint contiguous blocks
        blk = N // m
        if blk < 2:
            break
        h = [_ent_cols(v[b * blk:(b + 1) * blk, None])[0] for b in range(m)]
        H_list.append(float(np.mean(h)))
        N_list.append(blk)
    return _extrap(H_list, N_list, _ent_cols(v[:, None])[0])


def _extrap_trials(ids, n_fractions):
    """Noise entropy extrapolated to an infinite number of trials."""
    n = ids.shape[0]
    H_list, N_list = [], []
    for j in range(n_fractions):
        m = 2 ** j                                  # disjoint groups of trials
        sz = n // m
        if sz < 2:
            break
        h = [_ent_cols(ids[b * sz:(b + 1) * sz, :]).mean() for b in range(m)]
        H_list.append(float(np.mean(h)))
        N_list.append(sz)
    return _extrap(H_list, N_list, float(_ent_cols(ids).mean()))


def spike_entropy(bits, t_bin, L_list, overlap=True, extrapolate=True,
                  n_fractions=3):
    """Total and noise entropy of binarised spike trains (the direct method).

    bits         trials x bins 0/1 matrix from spike_bits
    t_bin        bin width                                            [ms]
    L_list       word lengths to try                                  [bins]
    overlap      take a word at every bin (True) or at every L-th bin
    extrapolate  correct the finite-sample bias by estimating the entropy on
                 fractions of the data and extrapolating to infinite data
    n_fractions  how many fractions (1, 1/2, 1/4, ...) to use

    Total entropy is the entropy of the word distribution pooled over every
    start time and every trial: how much variety the spike train has.

    Noise entropy holds the time fixed and looks across trials -- at each start
    time the words from the different trials form one distribution, its entropy
    is what the neuron does *not* pin down given the stimulus, and the average
    over start times is the noise entropy. This is the piece that needs many
    trials of the same frozen stimulus, and it is the piece that is zero when
    every trial is identical.

    Information = total - noise, extrapolated to 1/L -> 0 by the caller.

    Both are divided by the word duration and reported as rates in bits/s.
    """
    bits = np.asarray(bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    bits = (bits != 0).astype(np.int64)
    n_trials, n_bins = bits.shape

    L_list = np.atleast_1d(np.asarray(L_list, dtype=int))
    rate = 1000.0 * bits.sum() / (n_trials * n_bins * t_bin)

    nL = L_list.size
    out = {k: np.full(nL, np.nan) for k in
           ("H_total", "H_noise", "H_total_naive", "H_noise_naive")}
    n_words = np.zeros(nL, dtype=np.int64)
    n_seen = np.zeros(nL, dtype=np.int64)

    for m, L in enumerate(L_list):
        ids = _word_ids(bits, int(L), overlap)
        T = L * t_bin / 1000.0                       # word duration [s]

        # --- total entropy: every word, every trial, every start time ---
        pooled = ids.ravel()
        n_words[m] = pooled.size
        n_seen[m] = np.unique(pooled).size
        Ht_naive = float(_ent_cols(pooled[:, None])[0])
        Ht = _extrap_samples(pooled, n_fractions) if extrapolate else Ht_naive

        # --- noise entropy: across trials at fixed time, averaged over time --
        if n_trials > 1:
            Hn_naive = float(_ent_cols(ids).mean())
            Hn = (_extrap_trials(ids, n_fractions)
                  if extrapolate and n_trials >= 4 else Hn_naive)
        else:
            Hn_naive = Hn = np.nan

        out["H_total"][m] = Ht / T
        out["H_noise"][m] = Hn / T
        out["H_total_naive"][m] = Ht_naive / T
        out["H_noise_naive"][m] = Hn_naive / T

    return Entropy(L=L_list.astype(float), T=L_list * t_bin / 1000.0,
                   rate=rate, n_trials=n_trials, t_bin=t_bin,
                   info=out["H_total"] - out["H_noise"], n_words=n_words,
                   n_seen=n_seen, **out)


def shift_trials(bits, rng=None):
    """Circularly shift each trial by a random amount.

    A control for the direct method. The shift destroys the time-locking
    between trials, so the true information in the shifted raster is zero,
    while each trial keeps its own spike count and word statistics exactly.
    Whatever information the estimator still reports is its finite-sample
    floor for that number of trials and that word length.
    """
    if rng is None:
        rng = np.random.default_rng()
    bits = np.asarray(bits)
    out = np.empty_like(bits)
    for k in range(bits.shape[0]):
        out[k] = np.roll(bits[k], int(rng.integers(bits.shape[1])))
    return out
