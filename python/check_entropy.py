"""Check the entropy estimator against cases whose answer is known.

Mirrors check_entropy.m.

    python3 python/check_entropy.py        # a few seconds

Nothing here involves the neuron: these are synthetic spike trains chosen so
that the total entropy, the noise entropy, or the information is known exactly
in advance, which is the only way to tell an estimator bug from a real result.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lif import shift_trials, spike_bits, spike_entropy   # noqa: E402

T_BIN = 1.0                         # [ms]
P = 0.05                            # spike probability per bin -> 50 Hz


def H_bernoulli(p):
    """Entropy of one independent bin, in bits."""
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def check_binarisation():
    print("1. spike_bits: bin k covers ((k-1)*t_bin, k*t_bin]")
    st = np.array([0.4, 1.0, 1.6, 2.4, 9.9, 10.0])
    bits, merged = spike_bits([st], 10.0, 1.0)
    want = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    ok = np.array_equal(bits[0], want) and merged == 2
    print(f"   spikes {st} -> {bits[0]}, {merged} merged   [{'ok' if ok else 'FAILED'}]")
    print("   (0.4 and 1.0 share bin 1; 9.9 and 10.0 share bin 10)\n")
    return ok


def check_iid(rng):
    """Independent bins: the entropy rate is H(p)/t_bin at every word length.

    Compared against the entropy of the *empirical* p, which separates the
    estimator's error from the sampling error of p itself.
    """
    print("2. i.i.d. Bernoulli bins: entropy rate is H(p)/t_bin, independent of L")
    bits = (rng.random((1, 400_000)) < P).astype(np.uint8)
    truth = H_bernoulli(bits.mean()) / (T_BIN / 1000)     # empirical p
    L = np.array([1, 2, 4, 8, 12, 16])
    S = spike_entropy(bits, T_BIN, L)
    print(f"   true rate {truth:.2f} bits/s (empirical p = {bits.mean():.5f})")
    print("   L         :" + "".join(f"{v:9d}" for v in L))
    print("   naive  err:" + "".join(f"{v:+9.3f}" for v in S.H_total_naive - truth))
    print("   extrap err:" + "".join(f"{v:+9.3f}" for v in S.H_total - truth))
    worst = np.max(np.abs(S.H_total - truth))
    ok = worst < 0.005 * truth                     # 0.5% of the true rate
    print(f"   worst {worst:.3f} bits/s out of {truth:.0f}, tolerance 0.5%"
          f"   [{'ok' if ok else 'FAILED'}]\n")
    return ok


def check_periodic():
    """A period-p train has exactly p distinct words once L >= p-1."""
    print("3. period-7 train: exactly 7 distinct words, so log2(7) bits/word")
    bits = np.zeros((1, 21_000), dtype=np.uint8)
    bits[0, ::7] = 1
    L = np.array([2, 4, 6, 8, 12])
    S = spike_entropy(bits, T_BIN, L, extrapolate=False)
    per_word = S.H_total_naive * S.T
    print("   L        :" + "".join(f"{v:9d}" for v in L))
    print("   bits/word:" + "".join(f"{v:9.4f}" for v in per_word))
    ok = np.allclose(per_word[L >= 6], np.log2(7), atol=1e-6)
    print(f"   log2(7) = {np.log2(7):.4f} for L >= 6   [{'ok' if ok else 'FAILED'}]\n")
    return ok


def check_noise_entropy(rng):
    """Two limits: identical trials carry no noise, independent trials carry
    no information.

    The first is exact and is asserted. The second is only exact for an
    infinite number of trials, so what it really shows is the size of the
    finite-trial bias -- quantified in check 5.
    """
    print("4. the two limits of the noise entropy")
    L = np.array([2, 4, 8, 12])
    one = (rng.random((1, 20_000)) < P).astype(np.uint8)

    S = spike_entropy(np.repeat(one, 20, axis=0), T_BIN, L)
    max_noise = np.nanmax(np.abs(S.H_noise))
    ok_a = max_noise == 0
    print(f"   20 identical trials -> noise entropy {max_noise:.3g} bits/s, "
          f"information = total   [{'ok' if ok_a else 'FAILED'}]")

    bits = (rng.random((256, 20_000)) < P).astype(np.uint8)
    S = spike_entropy(bits, T_BIN, L)
    print("   256 independent trials -> true information is 0; what is left")
    print("   is the finite-trial bias, and it grows with the word length:")
    print("   L        :" + "".join(f"{v:9d}" for v in L))
    print(f"   info     :" + "".join(f"{v:9.1f}" for v in S.info)
          + f"   (total {S.H_total.max():.0f} bits/s)")
    print()
    return ok_a


def check_trial_count(rng):
    """How many trials the noise entropy needs, and whether the shift control
    sees the shortfall coming.

    Descriptive, not pass/fail: there is no threshold here that would not be
    arbitrary. The point is the scaling -- the bias falls roughly as 1/n_trials
    -- and that the control tracks it without knowing the answer.
    """
    print("5. finite-trial bias: residual information where the true value is 0")
    L = np.array([4, 8, 12, 16])
    bits = (rng.random((512, 20_000)) < P).astype(np.uint8)
    print("   trials |" + "".join(f"     L={v:<4d}" for v in L))
    for n in (8, 16, 32, 64, 128, 256, 512):
        S = spike_entropy(bits[:n], T_BIN, L)
        print(f"   {n:6d} |" + "".join(f"{v:10.1f}" for v in S.info))

    print("\n6. the shift control recovers that floor from time-locked data")
    one = (rng.random((1, 20_000)) < P).astype(np.uint8)
    for n in (16, 128):
        # Every trial locked to the same train, plus independent extra spikes.
        b = np.maximum(np.repeat(one, n, axis=0),
                       (rng.random((n, 20_000)) < 0.01).astype(np.uint8))
        S = spike_entropy(b, T_BIN, L)
        C = spike_entropy(shift_trials(b, rng), T_BIN, L)
        print(f"   n={n:<4d} info  :" + "".join(f"{v:10.1f}" for v in S.info))
        print("          floor :" + "".join(f"{v:10.1f}" for v in C.info))
    print()
    return True


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    results = [check_binarisation(), check_iid(rng), check_periodic(),
               check_noise_entropy(rng), check_trial_count(rng)]
    print("all checks passed" if all(results) else "SOME CHECKS FAILED")
