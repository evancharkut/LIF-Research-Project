% LIF_FILTERED_NOISE  Frozen-noise drive: repeated trials of the same stimulus.
%
% Generates one band-limited noise stimulus, presents it on every trial (the
% "frozen" stimulus), and adds a small independent noise current on each trial
% so that spike timing varies from trial to trial.
%
% This is the raster the jitter / reliability analysis is built on: spikes are
% grouped into events, reliability is the fraction of trials containing an
% event, and jitter is the standard deviation of the spike times within it.

rng(1);                       % reproducible stimulus and trial noise

dt       = 0.05;              % [ms]
duration = 1;                 % [s]
t_end    = duration * 1000;   % [ms]

n_trials  = 30;
rms_stim  = 1.0;              % [nA] rms of the frozen fluctuating component
dc_offset = 1.5;              % [nA] sits at rheobase
f_cut     = 40;               % [Hz]
rms_trial = 0.15;             % [nA] independent per-trial noise

% Frozen stimulus: generated once, reused on every trial.
[I_frozen, t_vect] = filtered_noise(t_end, dt, rms_stim, dc_offset, f_cut);

spikes_by_trial = cell(1, n_trials);
for k = 1:n_trials
    I_trial = filtered_noise(t_end, dt, rms_trial, 0, f_cut);
    [~, spikes_by_trial{k}] = lif_run(I_frozen + I_trial, dt);
end

n_spikes = cellfun(@numel, spikes_by_trial);
fprintf('%d trials, %.1f +/- %.1f spikes per trial (%.1f Hz mean rate)\n', ...
        n_trials, mean(n_spikes), std(n_spikes), mean(n_spikes)/duration);

p      = lif_params();
I_rheo = (p.V_thresh - p.E_rest)/p.R_m;

figure(3); clf
subplot(3,1,1)
plot(t_vect, I_frozen); hold on
yline(I_rheo, 'k--', 'rheobase');
ylabel('Current (nA)')
title(sprintf('Frozen filtered noise (%g Hz low-pass, %.1f nA rms + %.1f nA DC)', ...
              f_cut, rms_stim, dc_offset))
set(gca,'xlim',[0 t_end])

subplot(3,1,[2 3])
hold on
for k = 1:n_trials
    st = spikes_by_trial{k};
    if ~isempty(st)
        plot([st; st], [k-0.4; k+0.4]*ones(1,numel(st)), 'k', 'LineWidth', 0.6);
    end
end
xlabel('Time (ms)'); ylabel('Trial')
title('Spike raster across trials')
set(gca,'xlim',[0 t_end],'ylim',[0.5 n_trials+0.5])
