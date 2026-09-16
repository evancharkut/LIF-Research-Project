function [bits, n_merged] = spike_bits(spikes_by_trial, t_end, t_bin)
%SPIKE_BITS  Binarise spike trains into a trials x bins 0/1 matrix.
%
%   [bits, n_merged] = spike_bits(spikes_by_trial, t_end, t_bin)
%
%   spikes_by_trial  1 x n_trials cell array of spike-time vectors, or a
%                    single vector for a one-trial run               [ms]
%   t_end            trial duration                                  [ms]
%   t_bin            bin width                             (default 1) [ms]
%
%   bits      n_trials x floor(t_end/t_bin) logical array. Bin k covers
%             ((k-1)*t_bin, k*t_bin], matching CEIL(t/t_bin); a trailing
%             partial bin is dropped.
%   n_merged  number of spikes that landed in an already-occupied bin.
%
%   The direct method of Strong et al. reads each row as a binary word, which
%   assumes a bin is small enough to hold at most one spike. With an absolute
%   refractory period of t_ref the safe choice is t_bin <= t_ref, and
%   n_merged > 0 says that choice has been broken -- the words then under-
%   count spikes and both entropies come out too low. It is reported rather
%   than silently swallowed, which is what `bits(N,spike_bits) = 1` does.

if nargin < 3 || isempty(t_bin), t_bin = 1; end
if isnumeric(spikes_by_trial), spikes_by_trial = {spikes_by_trial}; end

n_trials = numel(spikes_by_trial);
n_bins   = floor(t_end / t_bin);
if n_bins < 1
    error('spike_bits:binTooWide', ...
          't_bin (%g ms) is larger than t_end (%g ms).', t_bin, t_end);
end

bits     = false(n_trials, n_bins);
n_merged = 0;
for k = 1:n_trials
    st  = spikes_by_trial{k}(:).';
    st  = st(st > 0 & st <= n_bins*t_bin);
    idx = max(1, ceil(st / t_bin));
    n_merged = n_merged + (numel(idx) - numel(unique(idx)));
    bits(k, idx) = true;
end

if n_merged > 0 && nargout < 2
    warning('spike_bits:multipleSpikesPerBin', ...
            ['%d spike(s) shared a bin with another at t_bin = %g ms. ' ...
             'Choose t_bin <= t_ref so that each bin holds at most one ' ...
             'spike.'], n_merged, t_bin);
end
end
