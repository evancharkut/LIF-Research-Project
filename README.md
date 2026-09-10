# Leaky integrate-and-fire neuron — spike timing and reliability

A MATLAB implementation of a leaky integrate-and-fire (LIF) neuron, built to study
how precisely a spiking neuron encodes a time-varying input: spike-timing jitter,
trial-to-trial reliability, and information transfer.

The model is driven by DC steps, sine waves, and band-limited ("filtered") noise,
and the simulation is validated against the closed-form firing-rate curve.

## The model

The membrane potential follows

```
tau * dV/dt = (E_rest - V) + I * R_m
```

with a hard threshold, a reset, and an absolute refractory period during which `V`
is clamped at `V_reset`.

| symbol | value | note |
|---|---|---|
| `E_rest` | -65 mV | resting potential |
| `V_thresh` | -50 mV | spike threshold |
| `V_reset` | -70 mV | post-spike reset |
| `R_m` | 10 MOhm | membrane resistance |
| `C_m` | 1 nF | membrane capacitance |
| `tau = R_m*C_m` | 10 ms | membrane time constant |
| `t_ref` | 3 ms | absolute refractory period |

Units are self-consistent throughout: I (nA) x R_m (MOhm) = mV, and MOhm x nF = ms.

The integration step is **exponential Euler**,

```matlab
V_inf = I*R_m + E_rest;
V     = V_inf + (V - V_inf)*exp(-dt/tau);
```

which is the exact solution for a current held constant across the step. This
matters: the simulated rate is already converged at `dt = 0.1 ms`, where a forward
Euler step of the same size still carries visible error.

## Validation

For a suprathreshold DC input the inter-spike interval has a closed form:

```
ISI = t_ref + tau * ln[(I*R_m + E_rest - V_reset) / (I*R_m + E_rest - V_thresh)]
```

which gives a **rheobase** (minimum current that fires the cell) of
`(V_thresh - E_rest)/R_m = 1.5 nA`, and a firing rate that saturates at
`1/t_ref = 333 Hz`.

Running `lif_fI_curve` at `dt = 0.05 ms` reproduces this to **within 1.6 Hz across
2-10 nA** — at 2 nA the simulation gives 52.0 Hz against a predicted 52.4 Hz. The
simulated cell is silent at 1.49 nA and fires at 1.51 nA, matching the analytic
rheobase, and driving it far past saturation gives 328 Hz against the 333 Hz
ceiling.

The residual error is threshold-crossing quantisation, not integration error: the
crossing is only detected at the end of a step, so each ISI is biased upward by up
to one `dt`. That is a fixed time error, so it costs proportionally more at high
rates — at 10 nA the gap shrinks from 1.4 Hz at `dt = 0.05 ms` to 0.03 Hz at
`dt = 0.002 ms`.

![f-I curve](figures/fI_curve.png)

![dt convergence](figures/dt_convergence.png)

## Frozen-noise raster

For the timing analysis the same band-limited noise stimulus is presented on every
trial, with a small independent noise current added per trial. Spike times cluster
into repeatable "events" where the stimulus rises steeply, and smear out where it
drifts slowly near threshold.

![frozen noise raster](figures/frozen_noise_raster.png)

## Jitter and reliability

Spikes pooled across trials are grouped into **events**: a peri-stimulus time
histogram bin holding at least 10% of the trials seeds an event, which is then
grown outwards over contiguous non-empty bins. For each event,

- **reliability** = fraction of trials that contributed a spike,
- **jitter** = standard deviation of the spike times within it.

An event in which any single trial fired more than once is discarded rather than
counted — its spread would measure the interval between two spikes instead of the
trial-to-trial jitter of one. Events below 50% reliability are still returned
per-event but left out of the summary means.

`lif_jitter_reliability` sweeps the leak conductance (`R_m = R_0/leak`, so
`leak = 1` is the default cell) and reports rate, jitter and reliability at each
value.

The event-finding rule is the one from Billimoria et al. (2006), lifted from an
earlier implementation. Two things were changed in the port: the "fired twice in
one event" test now uses a recorded trial index per spike rather than matching
floating-point spike times with `intersect`, and `hist` was replaced by
`histcounts`/`discretize`.

Validated against synthetic rasters with known event times, jitter and
reliability: recovered reliabilities are exact, and recovered jitter matches the
sample standard deviation of the spikes drawn into each event. Note that jitter
from 10 trials is a noisy statistic — the standard deviation of a sample standard
deviation at *n* = 10 is `sigma/sqrt(2(n-1))`, about 0.19 ms for a 0.8 ms event —
which is why the original averaged over repeated simulations.

![jitter and reliability](figures/jitter_reliability.png)

## Files

| file | what it does |
|---|---|
| `lif_params.m` | model constants, in one place so the simulation and the analytic check cannot drift apart |
| `lif_run.m` | core simulation — takes any current vector, returns `V(t)` and spike times |
| `lif_dc.m` | step response, with the measured rate printed |
| `lif_fI_curve.m` | firing rate vs. DC amplitude, with the analytic overlay |
| `filtered_noise.m` | stimulus generator: white noise, 4th-order Butterworth low-pass, scaled to a target rms with a DC offset |
| `lif_filtered_noise.m` | frozen-noise drive, spike raster across repeated trials |
| `spike_events.m` | groups spikes across trials into events; jitter and reliability per event |
| `lif_jitter_reliability.m` | jitter and reliability vs. leak conductance |
| `make_figures.m` | regenerates the figures in `figures/` |
| `check_setup.m` | preflight: shadowed built-ins, toolbox, duplicate files on the path |
| `python/` | one-for-one Python port; regenerates `figures/` without MATLAB |

## Running it

MATLAB R2020b or later. `filtered_noise.m` uses `butter` and `filtfilt` from the
Signal Processing Toolbox; `make_figures.m` uses `exportgraphics`.

```matlab
cd LIF-Research-Project
check_setup         % verify the path and toolbox first
lif_dc              % step response
lif_fI_curve        % validation against theory
lif_filtered_noise  % frozen-noise raster
lif_jitter_reliability  % jitter/reliability vs. leak
make_figures        % regenerate figures/
```

The figures in this README were produced by the Python port, which needs only
NumPy, SciPy and Matplotlib:

```bash
python3 python/make_figures.py
```

It reproduces every deterministic result above exactly; see
[python/README.md](python/README.md) for the comparison and for the two places
the two languages cannot agree.

## Notes on the original code

The simulation loop was rewritten from a first version that had several bugs worth
recording:

- the spike was written to `V_vect(i+1)` instead of `V_vect(i)`, leaving the true
  spike index at its initialised value of 0 mV and producing a spurious drop to
  0 mV at every spike — and growing the array past `length(t_vect)` on the last
  iteration;
- recorded spike times were offset by one step from the stimulus index;
- a `refract_flag` was set in two places and never read;
- during the refractory period `V` was left wherever it happened to be instead of
  being clamped to `V_reset`;
- the stimulus generator used `input` as a variable name, shadowing the MATLAB
  builtin.

In the rewrite the integration state is a scalar that never sees the cosmetic
spike height, so the drawn spike cannot feed back into the dynamics.

## In progress

- **Information transfer** — the direct method of Strong et al. (1998): binarise
  spike trains into words of length L, take total entropy across all trials and
  noise entropy from across-trial variability at fixed time, and take the
  difference.
- **A derivative-sensitive variant** of the model, to test the expectation that
  sharp upward stimulus slopes produce low-jitter, high-reliability spikes while
  slow ramps near threshold produce high jitter and dropped spikes.

## References

- Strong, S.P., Koberle, R., de Ruyter van Steveninck, R.R., Bialek, W. (1998).
  Entropy and information in neural spike trains. *Physical Review Letters* 80(1),
  197-200.
- Billimoria, C.P., DiCaprio, R.A., Birmingham, J.T., Abbott, L.F., Marder, E.
  (2006). Neuromodulation of spike-timing precision in sensory neurons. *Journal
  of Neuroscience* 26(22), 5910-5919.

## License

MIT — see [LICENSE](LICENSE).
