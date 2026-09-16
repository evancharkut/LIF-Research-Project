function S = spike_entropy(bits, t_bin, L_list, opts)
%SPIKE_ENTROPY  Total and noise entropy of binarised spike trains.
%
%   S = spike_entropy(bits, t_bin, L_list, opts)
%
%   The direct method of Strong et al. (1998). The spike train is cut into
%   binary words of L bins, and two entropies are taken of the resulting word
%   distributions:
%
%   TOTAL entropy is the entropy of the words pooled over every start time and
%   every trial -- the whole variety the spike train is capable of. One long
%   stimulus and a single trial is enough to measure it.
%
%   NOISE entropy holds the time fixed and looks *across* trials: at each
%   start time the words from the different trials form one distribution,
%   whose entropy is what the neuron leaves undetermined when the stimulus is
%   already fixed. Averaging that over start times gives the noise entropy.
%   This is the piece that needs many trials of the same frozen stimulus, and
%   the piece that is identically zero when every trial is a copy.
%
%   INFORMATION = total - noise, extrapolated to 1/L -> 0.
%
%   bits     n_trials x n_bins 0/1 matrix, from SPIKE_BITS
%   t_bin    bin width                                               [ms]
%   L_list   word lengths to try                                     [bins]
%   opts   (optional) struct; any field left out takes the default:
%     .overlap      take a word starting at every bin (true) or at every
%                   L-th bin, i.e. non-overlapping words      (true)
%     .extrapolate  correct the finite-sample bias: re-estimate on 1/2 and
%                   1/4 of the data and extrapolate to infinite data (true)
%     .n_fractions  how many fractions to use                (3)
%
%   S.L, S.T           word length [bins] and duration [s]
%   S.rate             mean firing rate                              [Hz]
%   S.H_total          total entropy rate                            [bits/s]
%   S.H_noise          noise entropy rate, NaN for a single trial     [bits/s]
%   S.info             S.H_total - S.H_noise                          [bits/s]
%   S.H_total_naive    the same before the finite-sample correction
%   S.H_noise_naive
%   S.n_words          words behind each total-entropy estimate
%   S.n_seen           distinct words actually observed
%
%   Rates are per second; divide by S.rate for bits per spike.
%
%   Two cautions. The noise entropy is estimated from n_trials samples per
%   start time, so it is the biased one: with too few trials it comes out too
%   low and the information too high. SHIFT_TRIALS gives the size of that
%   error directly. And a plug-in entropy can only ever see min(2^L, samples)
%   distinct words, so L must stay well below log2(number of samples) --
%   S.n_seen against 2^L is the check.

if nargin < 3 || isempty(L_list), L_list = 1:10; end
if nargin < 4, opts = struct(); end
d = struct('overlap', true, 'extrapolate', true, 'n_fractions', 3);
f = fieldnames(d);
for k = 1:numel(f)
    if ~isfield(opts, f{k}), opts.(f{k}) = d.(f{k}); end
end

bits = double(bits ~= 0);
[n_trials, n_bins] = size(bits);
if any(L_list > n_bins)
    error('spike_entropy:wordTooLong', ...
          'Word length %d exceeds the %d bins available.', ...
          max(L_list), n_bins);
end
if any(L_list > 52)
    % Each word is packed into one double, which is exact only to 2^53. Word
    % lengths anywhere near this are hopeless to sample anyway.
    error('spike_entropy:wordTooLongToPack', ...
          'Word length %d cannot be packed into a double exactly.', ...
          max(L_list));
end

nL = numel(L_list);
S  = struct('L', L_list(:).', 'T', L_list(:).'*t_bin/1000, ...
            'rate', 1000*sum(bits(:))/(n_trials*n_bins*t_bin), ...
            'n_trials', n_trials, 't_bin', t_bin, ...
            'H_total', nan(1,nL), 'H_noise', nan(1,nL), 'info', nan(1,nL), ...
            'H_total_naive', nan(1,nL), 'H_noise_naive', nan(1,nL), ...
            'n_words', zeros(1,nL), 'n_seen', zeros(1,nL));

for m = 1:nL
    L   = L_list(m);
    ids = word_ids(bits, L, opts.overlap);     % n_trials x n_positions
    T   = L * t_bin / 1000;                    % word duration [s]

    % --- total entropy: every word, every trial, every start time --------
    pooled        = ids(:);
    S.n_words(m)  = numel(pooled);
    S.n_seen(m)   = numel(unique(pooled));
    Ht_naive      = ent_cols(pooled);
    if opts.extrapolate
        Ht = extrap_samples(pooled, opts.n_fractions);
    else
        Ht = Ht_naive;
    end

    % --- noise entropy: across trials at fixed time, averaged over time ---
    if n_trials > 1
        Hn_naive = mean(ent_cols(ids));
        if opts.extrapolate && n_trials >= 4
            Hn = extrap_trials(ids, opts.n_fractions);
        else
            Hn = Hn_naive;
        end
    else
        Hn_naive = NaN;
        Hn       = NaN;
    end

    S.H_total(m)       = Ht       / T;
    S.H_noise(m)       = Hn       / T;
    S.H_total_naive(m) = Ht_naive / T;
    S.H_noise_naive(m) = Hn_naive / T;
end

S.info = S.H_total - S.H_noise;
end

% -----------------------------------------------------------------------
function ids = word_ids(bits, L, overlap)
% Each length-L binary word packed into one integer, trials x start positions.
n_pos = size(bits,2) - L + 1;
ids   = zeros(size(bits,1), n_pos);
for k = 1:L
    ids = ids*2 + bits(:, k:k+n_pos-1);
end
if ~overlap
    ids = ids(:, 1:L:end);
end
end

% -----------------------------------------------------------------------
function H = ent_cols(X)
% Plug-in entropy of each column of X, in bits.
%
% Each of the c members of a group of identical entries contributes
% -(1/n)*log2(c/n), so the group as a whole contributes -(c/n)*log2(c/n),
% which is the plug-in sum -sum(p.*log2(p)). Summing over elements rather
% than over groups is what lets every column be done at once, without the
% UNIQUE(...,'rows') call the word-by-word version needs.
[n, m] = size(X);
if n < 2, H = zeros(1, m); return, end

Z = sort(X, 1);
g = cumsum([true(1,m); diff(Z,1,1) ~= 0], 1);        % group index in column
clear Z
lin = g + (0:m-1)*n;                                 % into an n x m grid
clear g
counts = accumarray(lin(:), 1, [n*m, 1]);
c = reshape(counts(lin), n, m);                      % group size of each entry
H = -sum(log2(c/n), 1) / n;
end

% -----------------------------------------------------------------------
function H = extrap_samples(v, n_fractions)
% Total entropy extrapolated to an infinite number of words.
%
% A plug-in entropy estimated from N samples is biased low by roughly
% (K-1)/(2*N*log(2)) for K occupied words. Re-estimating on 1/2 and 1/4 of the
% data and fitting a straight line in 1/N removes that leading term, which is
% the correction in Strong et al. (1998).
N = numel(v);
H_list = []; N_list = [];
for j = 1:n_fractions
    blocks = 2^(j-1);                     % disjoint contiguous blocks
    blk    = floor(N/blocks);
    if blk < 2, break, end
    h = zeros(1, blocks);
    for b = 1:blocks
        h(b) = ent_cols(v((b-1)*blk + (1:blk).'));
    end
    H_list(end+1) = mean(h);   %#ok<AGROW>
    N_list(end+1) = blk;       %#ok<AGROW>
end
H = intercept(H_list, N_list, ent_cols(v));
end

% -----------------------------------------------------------------------
function H = extrap_trials(ids, n_fractions)
% Noise entropy extrapolated to an infinite number of trials.
n = size(ids,1);
H_list = []; N_list = [];
for j = 1:n_fractions
    groups = 2^(j-1);                     % disjoint groups of trials
    sz     = floor(n/groups);
    if sz < 2, break, end
    h = zeros(1, groups);
    for b = 1:groups
        h(b) = mean(ent_cols(ids((b-1)*sz + (1:sz), :)));
    end
    H_list(end+1) = mean(h);   %#ok<AGROW>
    N_list(end+1) = sz;        %#ok<AGROW>
end
H = intercept(H_list, N_list, mean(ent_cols(ids)));
end

% -----------------------------------------------------------------------
function H = intercept(H_list, N_list, H_full)
% Value a straight line through entropy vs. 1/sample-count takes at 1/N = 0.
if numel(H_list) < 2
    H = H_full;
    return
end
c = polyfit(1./N_list, H_list, 1);
H = c(2);
end
