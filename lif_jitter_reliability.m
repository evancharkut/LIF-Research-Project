% LIF_JITTER_RELIABILITY  Spike-timing precision vs. leak conductance.
%
% Frozen-noise drive at a range of leak conductances. The membrane resistance
% is scaled as R_m = R_0/leak, so "leak = 1" is the default cell in
% LIF_PARAMS and larger values are leakier: shorter tau, and less excitable
% for the same injected current.
%
% For each leak value the same frozen stimulus is presented on every trial
% with independent noise added per trial, spikes are grouped into events by
% SPIKE_EVENTS, and the mean jitter and reliability of those events are
% reported.
%
% Defaults follow the parameters in the original intandfirejit2026 dialog:
% 6 s, 10 trials, 3 ms refractory, 0.5 ms bins, 50% event threshold.

rng(2);                        % reproducible stimulus and trial noise

dt       = 0.05;               % [ms]
duration = 6;                  % [s]
t_end    = duration * 1000;    % [ms]
n_trials = 10;
leak     = 1:8;                % leak normalised to the LIF_PARAMS value

sig_rms    = 5;                % [nA] rms of the frozen signal
sig_fcut   = 100;              % [Hz] low-pass cutoff of the signal
noise_rms  = 1;                % [nA] rms of the independent per-trial noise
noise_fcut = 20;               % [Hz] low-pass cutoff of that noise
dc         = 0;                % [nA]

opts = struct('binwidth', 0.5, 't_start', 1000, ...
              'peak_frac', 0.10, 'rel_thresh', 0.50);

p0  = lif_params();
R_0 = p0.R_m;

rate = zeros(size(leak));
jit  = zeros(size(leak));
rel  = zeros(size(leak));
nev  = zeros(size(leak));

for m = 1:numel(leak)
    p     = p0;
    p.R_m = R_0 / leak(m);

    % Frozen stimulus: generated once per leak value, reused on every trial.
    I_frozen = filtered_noise(t_end, dt, sig_rms, dc, sig_fcut);

    spikes_by_trial = cell(1, n_trials);
    for tr = 1:n_trials
        I_noise = filtered_noise(t_end, dt, noise_rms, 0, noise_fcut);
        [~, spikes_by_trial{tr}] = lif_run(I_frozen + I_noise, dt, p);
    end

    ev      = spike_events(spikes_by_trial, t_end, opts);
    rate(m) = ev.rate;
    jit(m)  = ev.jitter_mean;
    rel(m)  = ev.reliability_mean;
    nev(m)  = ev.n_events;

    fprintf(['leak %g (tau %4.2f ms): %5.1f Hz, %3d events ' ...
             '(%d discarded), jitter %.3f ms, reliability %.2f\n'], ...
            leak(m), p.R_m*p.C_m, ev.rate, ev.n_events, ev.n_multiple, ...
            ev.jitter_mean, ev.reliability_mean);
end

figure(4); clf
subplot(3,1,1)
plot(leak, rate, 'o-'); ylabel('Rate (Hz)'); grid on
title(sprintf(['Timing precision vs. leak (%d trials, %g s, ' ...
               '%g nA rms signal + %g nA rms noise)'], ...
              n_trials, duration, sig_rms, noise_rms))
subplot(3,1,2)
plot(leak, jit, 'o-'); ylabel('Jitter (ms)'); grid on
subplot(3,1,3)
plot(leak, rel, 'o-'); ylabel('Reliability'); xlabel('Leak (x default)')
set(gca, 'ylim', [0 1]); grid on
