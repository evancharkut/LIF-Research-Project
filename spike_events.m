function ev = spike_events(spikes_by_trial, t_end, opts)
%SPIKE_EVENTS  Group spikes across trials into events; jitter and reliability.
%
%   ev = spike_events(spikes_by_trial, t_end, opts)
%
%   Pools the spikes from every trial into one peri-stimulus time histogram,
%   finds the repeatable "events" in it, and measures how precisely and how
%   dependably each event is fired.
%
%   spikes_by_trial  1 x n_trials cell array of spike-time vectors  [ms]
%   t_end            trial duration                                 [ms]
%   opts   (optional) struct; any field left out takes the default:
%     .binwidth    PSTH bin width                          (0.5)  [ms]
%     .t_start     ignore spikes before this, to skip the transient
%                  while the cell settles                  (1000) [ms]
%     .peak_frac   a bin holding at least this fraction of the trials seeds
%                  an event                                (0.10)
%     .rel_thresh  events below this reliability are left out of the summary
%                  means, but are still returned per-event (0.50)
%
%   ev.times         event centre, the mean spike time within it  [ms]
%   ev.jitter        s.d. of the spike times within the event     [ms]
%   ev.reliability   fraction of trials that contributed a spike
%   ev.jitter_mean       mean jitter      over events above rel_thresh  [ms]
%   ev.reliability_mean  mean reliability over events above rel_thresh
%   ev.n_multiple    events discarded because some trial fired twice inside
%   ev.rate          pooled firing rate over the counted window   [Hz]
%
%   The event-finding rule is taken from Billimoria et al. (2006): a bin busy
%   enough to be a peak is grown outwards over contiguous non-empty bins, and
%   the resulting run of bins is one event. An event in which any single trial
%   fired more than once is thrown away rather than counted, since its spread
%   would measure the interval between two spikes rather than the trial-to-
%   trial jitter of one.

if nargin < 3, opts = struct(); end
d = struct('binwidth', 0.5, 't_start', 1000, 'peak_frac', 0.10, ...
           'rel_thresh', 0.50);
f = fieldnames(d);
for k = 1:numel(f)
    if ~isfield(opts, f{k}), opts.(f{k}) = d.(f{k}); end
end

n_trials = numel(spikes_by_trial);

% --- pool the trials, remembering which trial each spike came from -------
% The trial index is what makes the "fired twice in one event" test exact;
% the original matched spike times against a padded array with INTERSECT,
% which relies on floating-point times comparing equal.
n_per   = cellfun(@(s) sum(s > opts.t_start), spikes_by_trial);
pooled   = zeros(1, sum(n_per));
trial_id = zeros(1, sum(n_per));
pos = 0;
for k = 1:n_trials
    st = spikes_by_trial{k}(:).';
    st = st(st > opts.t_start);
    n  = numel(st);
    pooled(pos+(1:n))   = st;
    trial_id(pos+(1:n)) = k;
    pos = pos + n;
end

ev = struct('times', [], 'jitter', [], 'reliability', [], ...
            'jitter_mean', NaN, 'reliability_mean', NaN, ...
            'n_multiple', 0, 'n_events', 0, 'rate', 0);

ev.rate = 1000 * numel(pooled) / ((t_end - opts.t_start) * n_trials);
if isempty(pooled), return, end

% --- peri-stimulus time histogram ---------------------------------------
edges = 0:opts.binwidth:t_end;
if edges(end) < t_end, edges(end+1) = edges(end) + opts.binwidth; end
n_bins = numel(edges) - 1;
X      = histcounts(pooled, edges);
bin_of = discretize(pooled, edges);

% --- grow each peak bin into an event ------------------------------------
peak_bins = find(X >= opts.peak_frac * n_trials);

ev_time = zeros(1, numel(peak_bins));
ev_jit  = zeros(1, numel(peak_bins));
ev_rel  = zeros(1, numel(peak_bins));
n_ev    = 0;
last_hi = 0;                       % last bin already absorbed into an event

for c = 1:numel(peak_bins)
    pb = peak_bins(c);
    if pb <= last_hi, continue, end     % already inside the previous event

    lo = pb; while lo > 1      && X(lo-1) > 0, lo = lo - 1; end
    hi = pb; while hi < n_bins && X(hi+1) > 0, hi = hi + 1; end
    last_hi = hi;

    in_ev  = bin_of >= lo & bin_of <= hi;
    times  = pooled(in_ev);
    trials = trial_id(in_ev);

    % Discard the event if any trial put more than one spike in it.
    if max(histcounts(trials, 0.5:1:(n_trials+0.5))) > 1
        ev.n_multiple = ev.n_multiple + 1;
        continue
    end

    n_ev = n_ev + 1;
    ev_time(n_ev) = mean(times);
    ev_jit(n_ev)  = std(times);
    ev_rel(n_ev)  = numel(times) / n_trials;
end

ev.times       = ev_time(1:n_ev);
ev.jitter      = ev_jit(1:n_ev);
ev.reliability = ev_rel(1:n_ev);
ev.n_events    = n_ev;

good = ev.reliability > opts.rel_thresh;
if any(good)
    ev.jitter_mean      = mean(ev.jitter(good));
    ev.reliability_mean = mean(ev.reliability(good));
end
end
