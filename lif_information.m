% LIF_INFORMATION  Information carried by the spike train: the direct method.
%
% Information = total entropy - noise entropy (Strong et al. 1998).
%
% Both are measured on binary words of L bins cut out of the spike train, and
% both are entropy *rates*, so each is divided by the word duration. The two
% halves need different experiments:
%
%   total entropy  how much variety the spike train has at all. One trial of a
%                  long stimulus, so that the word distribution is well
%                  sampled.
%   noise entropy  how much of that variety survives when the stimulus is
%                  already known. Many trials of the same frozen stimulus:
%                  the words at one fixed time, taken across trials, are all
%                  responses to the same input, so their spread is noise.
%
% Neither entropy has converged at any finite L -- words too short miss the
% correlations between spikes. The estimate is the intercept of a straight
% line in 1/L, which is why the figure is plotted against inverse word length.
%
% Runs two simulations and takes about a minute.

rng(5);                        % reproducible stimulus and trial noise

dt      = 0.1;                 % [ms] converged; see the f-I validation
t_bin   = 1;                   % [ms] one bin holds at most one spike here,
                               %      since the refractory period is 3 ms
t_start = 1000;                % [ms] discard the settling transient
L_list  = [4 5 6 8 10 12 14 16];       % word lengths [bins]
L_fit   = [6 14];              % the range fitted for the 1/L -> 0 intercept

dc        = 1.5;               % [nA] sits at rheobase
rms_stim  = 1.0;               % [nA] rms of the fluctuating stimulus
f_cut     = 40;                % [Hz]
rms_trial = 0.15;              % [nA] independent per-trial noise

t_long   = 200 * 1000;         % [ms] single long trial, for the total entropy
t_trial  = 16 * 1000;          % [ms] frozen stimulus, repeated
n_trials = 128;                % enough that the shift control is near zero

% --- one long trial: total entropy ---------------------------------------
% The per-trial noise is present here too: the total entropy has to be the
% entropy of the responses the cell actually produces, noise included.
I_long = filtered_noise(t_long, dt, rms_stim, dc, f_cut) + ...
         filtered_noise(t_long, dt, rms_trial, 0, f_cut);
[~, spikes_long] = lif_run(I_long, dt);

[bits_long, merged_long] = spike_bits(spikes_long, t_long, t_bin);
bits_long = bits_long(:, t_start/t_bin+1:end);
S_long    = spike_entropy(bits_long, t_bin, L_list);

% --- many trials of a frozen stimulus: noise entropy ----------------------
I_frozen        = filtered_noise(t_trial, dt, rms_stim, dc, f_cut);
spikes_by_trial = cell(1, n_trials);
for k = 1:n_trials
    I_noise = filtered_noise(t_trial, dt, rms_trial, 0, f_cut);
    [~, spikes_by_trial{k}] = lif_run(I_frozen + I_noise, dt);
end

[bits_tr, merged_tr] = spike_bits(spikes_by_trial, t_trial, t_bin);
bits_tr = bits_tr(:, t_start/t_bin+1:end);
S_tr    = spike_entropy(bits_tr, t_bin, L_list);

% The control: the same trials with the time-locking shifted away, where the
% true information is zero. What comes back is the estimator's floor.
S_ctrl = spike_entropy(shift_trials(bits_tr), t_bin, L_list);
floor_ = S_ctrl.info;

fprintf('long run : %g s, 1 trial, %.2f Hz, %d merged spikes\n', ...
        t_long/1000, S_long.rate, merged_long);
fprintf('frozen   : %g s x %d trials, %.2f Hz, %d merged spikes\n', ...
        t_trial/1000, n_trials, S_tr.rate, merged_tr);
fprintf(['total entropy from the two runs agrees to %.1f%% at L = %d ' ...
         '(%.1f vs %.1f bits/s)\n'], ...
        100*abs(S_long.H_total(1)-S_tr.H_total(1))/S_tr.H_total(1), ...
        L_list(1), S_long.H_total(1), S_tr.H_total(1));

fprintf('\n  L   1/L   H_total  H_noise     info    floor   bits/spike  words seen\n');
for m = 1:numel(L_list)
    fprintf('%3d  %.3f  %7.1f  %7.1f  %7.1f  %7.1f   %8.2f   %6d/%d\n', ...
            L_list(m), 1/L_list(m), S_tr.H_total(m), S_tr.H_noise(m), ...
            S_tr.info(m), floor_(m), S_tr.info(m)/S_tr.rate, ...
            S_tr.n_seen(m), 2^L_list(m));
end

% --- extrapolate each entropy rate to 1/L -> 0 ----------------------------
x   = 1 ./ L_list;
in_fit = L_list >= L_fit(1) & L_list <= L_fit(2);   % "fit" would shadow FIT
c_t = polyfit(x(in_fit), S_tr.H_total(in_fit), 1);
c_n = polyfit(x(in_fit), S_tr.H_noise(in_fit), 1);
c_l = polyfit(x(in_fit), S_long.H_total(in_fit), 1);

H_total_inf = c_t(2);
H_noise_inf = c_n(2);
info_inf    = H_total_inf - H_noise_inf;

fprintf(['\nextrapolated to 1/L -> 0 over L = %d..%d:\n' ...
         '  total entropy %6.1f bits/s  (%.1f from the long run)\n' ...
         '  noise entropy %6.1f bits/s\n' ...
         '  information   %6.1f bits/s = %.2f bits/spike at %.1f Hz\n'], ...
        L_fit(1), L_fit(2), H_total_inf, c_l(2), H_noise_inf, ...
        info_inf, info_inf/S_tr.rate, S_tr.rate);

% --- figure ---------------------------------------------------------------
xf = [0 max(x)];

figure(5); clf
subplot(1,2,1)
plot(x, S_tr.H_total, 'o-', 'DisplayName', 'total (frozen trials)'); hold on
plot(x, S_long.H_total, 's--', 'DisplayName', 'total (one long trial)');
plot(x, S_tr.H_noise, 'o-', 'DisplayName', 'noise');
plot(xf, polyval(c_t, xf), 'k:', 'HandleVisibility', 'off');
plot(xf, polyval(c_n, xf), 'k:', 'HandleVisibility', 'off');
plot(0, H_total_inf, 'k*', 'HandleVisibility', 'off');
plot(0, H_noise_inf, 'k*', 'HandleVisibility', 'off');
xlabel('1 / word length'); ylabel('Entropy rate (bits/s)')
title(sprintf('Entropy vs. word length (%d trials, %g ms bins)', ...
              n_trials, t_bin))
legend('Location', 'east'); grid on; set(gca, 'xlim', xf)

subplot(1,2,2)
plot(x, S_tr.info, 'o-', 'DisplayName', 'information'); hold on
plot(x, floor_, 'x-', 'DisplayName', 'shift control (floor)');
plot(xf, polyval(c_t-c_n, xf), 'k:', 'HandleVisibility', 'off');
plot(0, info_inf, 'k*', 'HandleVisibility', 'off');
text(0.01, 0.93*info_inf, sprintf('%.0f bits/s\n%.2f bits/spike', ...
     info_inf, info_inf/S_tr.rate), 'VerticalAlignment', 'top');
xlabel('1 / word length'); ylabel('Information rate (bits/s)')
title('Total minus noise')
legend('Location', 'east'); grid on
set(gca, 'xlim', xf, 'ylim', [-0.06 1.2]*info_inf)
