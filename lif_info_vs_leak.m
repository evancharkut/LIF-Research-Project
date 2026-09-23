% LIF_INFO_VS_LEAK  Information, jitter and reliability vs. leak, averaged
% over many frozen stimuli.
%
% LIF_JITTER_RELIABILITY uses one frozen stimulus per leak value and 10
% trials: enough for jitter and reliability, not for the noise entropy, and
% every leak value sees a different input. Here both are fixed:
%
%   - n_stim independent frozen stimuli, each presented for n_trials trials,
%     so every point is a mean over stimuli with a standard error across them;
%   - every leak value sees the SAME n_stim stimuli and the SAME per-trial
%     noise, so the difference between two leak values is the leak and
%     nothing else (common random numbers).
%
% The stimulus is the one from LIF_JITTER_RELIABILITY, so jitter, reliability
% and information all come from the same rasters. The information is estimated
% as in LIF_INFORMATION: 1 ms words, extrapolated to 1/L -> 0 over L = 6..14,
% with the shift control run on every raster.
%
% Uses PARFOR when the Parallel Computing Toolbox is present. Takes several
% minutes; python/info_vs_leak.py is the same calculation.

seed     = 7;
n_stim   = 10;                 % independent frozen stimuli
n_trials = 100;                % trials of each
leak     = 1:8;                % leak normalised to the LIF_PARAMS value

dt      = 0.05;                % [ms]
t_trial = 11 * 1000;           % [ms] 1 s transient + 10 s analysed
t_start = 1000;                % [ms]

sig_rms    = 5;   sig_fcut   = 100;   % [nA], [Hz] frozen signal
noise_rms  = 1;   noise_fcut = 20;    % [nA], [Hz] per-trial noise
dc         = 0;                       % [nA]

t_bin   = 1;                   % [ms]
L_list  = [4 5 6 8 10 12 14 16];
L_fit   = [6 14];
L_floor = 14;

opts = struct('binwidth', 0.5, 't_start', t_start, ...
              'peak_frac', 0.10, 'rel_thresh', 0.50);

p0   = lif_params();
R_0  = p0.R_m;
x    = 1 ./ L_list;
in_fit = L_list >= L_fit(1) & L_list <= L_fit(2);
iL   = find(L_list == L_floor);

nL = numel(leak);
[rate, jit, rel, H_tot, H_noi, info, floor_] = deal(zeros(n_stim, nL));

parfor s = 1:n_stim
    % Seeds are a function of (stimulus, trial) only, so every leak value
    % sees exactly the same input on trial k of stimulus s.
    rng(seed*1e6 + s*1e3);
    I_frozen = filtered_noise(t_trial, dt, sig_rms, dc, sig_fcut);

    [r_s, j_s, rl_s, ht_s, hn_s, fl_s] = deal(zeros(1, nL));
    for m = 1:nL
        p     = p0;
        p.R_m = R_0 / leak(m);

        spikes_by_trial = cell(1, n_trials);
        for k = 1:n_trials
            rng(seed*1e6 + s*1e3 + k);
            I_noise = filtered_noise(t_trial, dt, noise_rms, 0, noise_fcut);
            [~, spikes_by_trial{k}] = lif_run(I_frozen + I_noise, dt, p);
        end

        ev   = spike_events(spikes_by_trial, t_trial, opts);
        bits = spike_bits(spikes_by_trial, t_trial, t_bin);
        bits = bits(:, t_start/t_bin+1:end);
        S    = spike_entropy(bits, t_bin, L_list);
        rng(seed*1e6 + s*1e3 + 999);
        S_ctrl = spike_entropy(shift_trials(bits), t_bin, L_list);

        c_t = polyfit(x(in_fit), S.H_total(in_fit), 1);
        c_n = polyfit(x(in_fit), S.H_noise(in_fit), 1);

        r_s(m)  = S.rate;
        j_s(m)  = ev.jitter_mean;
        rl_s(m) = ev.reliability_mean;
        ht_s(m) = c_t(2);
        hn_s(m) = c_n(2);
        fl_s(m) = S_ctrl.info(iL);
    end
    rate(s,:) = r_s;  jit(s,:) = j_s;   rel(s,:)   = rl_s;
    H_tot(s,:) = ht_s; H_noi(s,:) = hn_s; floor_(s,:) = fl_s;
end
info = H_tot - H_noi;
bps  = info ./ rate;

sem = @(X) std(X, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(X), 1));
avg = @(X) mean(X, 1, 'omitnan');

fprintf(['%d stimuli x %d trials x %g s, %g ms bins, fit L = %d..%d; ' ...
         'mean +/- s.e.m.\n\n'], n_stim, n_trials, (t_trial-t_start)/1000, ...
        t_bin, L_fit(1), L_fit(2));
fprintf(['leak   rate (Hz)       jitter (ms)     reliability     ' ...
         'info (bits/s)   bits/spike      floor@L%d\n'], L_floor);
for m = 1:nL
    fprintf(['%3d   %5.1f +/- %4.1f  %5.3f +/- %5.3f  %5.3f +/- %5.3f  ' ...
             '%5.1f +/- %4.1f  %5.2f +/- %4.2f  %5.1f\n'], leak(m), ...
            avg(rate(:,m)), sem(rate(:,m)), avg(jit(:,m)), sem(jit(:,m)), ...
            avg(rel(:,m)), sem(rel(:,m)), avg(info(:,m)), sem(info(:,m)), ...
            avg(bps(:,m)), sem(bps(:,m)), avg(floor_(:,m)));
end

% --- figure: one thin grey line per stimulus, mean +/- s.e.m. on top ------
panels = {rate, 'Rate (Hz)'; jit, 'Jitter (ms)'; rel, 'Reliability'; ...
          info, 'Information (bits/s)'; bps, 'Information (bits/spike)'};
figure(6); clf
for q = 1:size(panels, 1)
    X = panels{q, 1};
    subplot(1, 5, q)
    plot(leak, X', 'Color', [0.75 0.75 0.75]); hold on
    errorbar(leak, avg(X), sem(X), 'o-', 'LineWidth', 2, 'CapSize', 4)
    xlabel('Leak (x default)'); ylabel(panels{q, 2}); grid on
    if q == 3, set(gca, 'ylim', [0 1]); end
    if q >= 4, set(gca, 'ylim', [0 max(ylim)]); end
end
sgtitle(sprintf(['Leak sweep, averaged over %d frozen stimuli x %d ' ...
                 'trials (grey: each stimulus; blue: mean \\pm s.e.m.)'], ...
                n_stim, n_trials))
