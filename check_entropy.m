% CHECK_ENTROPY  Check the entropy estimator against cases with known answers.
%
% Nothing here involves the neuron: these are synthetic spike trains chosen so
% that the total entropy, the noise entropy or the information is known exactly
% in advance, which is the only way to tell an estimator bug from a real
% result. Mirrors python/check_entropy.py.
%
%   check_entropy        % a few seconds

rng(0);

t_bin = 1;                    % [ms]
p     = 0.05;                 % spike probability per bin -> 50 Hz
ok    = true(1,4);

% --- 1. binarisation ------------------------------------------------------
fprintf('1. spike_bits: bin k covers ((k-1)*t_bin, k*t_bin]\n');
st = [0.4 1.0 1.6 2.4 9.9 10.0];
[bits, merged] = spike_bits(st, 10, 1);
want  = logical([1 1 1 0 0 0 0 0 0 1]);
ok(1) = isequal(bits, want) && merged == 2;
fprintf('   spikes -> [%s], %d merged   [%s]\n', ...
        num2str(double(bits)), merged, pass(ok(1)));
fprintf('   (0.4 and 1.0 share bin 1; 9.9 and 10.0 share bin 10)\n\n');

% --- 2. independent bins: entropy rate is H(p)/t_bin at every L -----------
% Compared against the entropy of the *empirical* p, which separates the
% estimator's error from the sampling error of p itself.
fprintf('2. i.i.d. Bernoulli bins: entropy rate is H(p)/t_bin, independent of L\n');
bits  = rand(1, 400000) < p;
p_hat = mean(bits);
truth = -(p_hat*log2(p_hat) + (1-p_hat)*log2(1-p_hat)) / (t_bin/1000);
L     = [1 2 4 8 12 16];
S     = spike_entropy(bits, t_bin, L);
fprintf('   true rate %.2f bits/s (empirical p = %.5f)\n', truth, p_hat);
fprintf('   L         :%s\n', sprintf('%9d', L));
fprintf('   naive  err:%s\n', sprintf('%+9.3f', S.H_total_naive - truth));
fprintf('   extrap err:%s\n', sprintf('%+9.3f', S.H_total - truth));
worst = max(abs(S.H_total - truth));
ok(2) = worst < 0.005*truth;                       % 0.5% of the true rate
fprintf('   worst %.3f bits/s out of %.0f, tolerance 0.5%%   [%s]\n\n', ...
        worst, truth, pass(ok(2)));

% --- 3. a period-P train has exactly P distinct words once L >= P-1 -------
fprintf('3. period-7 train: exactly 7 distinct words, so log2(7) bits/word\n');
bits = false(1, 21000);
bits(1:7:end) = true;
L = [2 4 6 8 12];
S = spike_entropy(bits, t_bin, L, struct('extrapolate', false));
per_word = S.H_total_naive .* S.T;
fprintf('   L        :%s\n', sprintf('%9d', L));
fprintf('   bits/word:%s\n', sprintf('%9.4f', per_word));
ok(3) = all(abs(per_word(L >= 6) - log2(7)) < 1e-6);
fprintf('   log2(7) = %.4f for L >= 6   [%s]\n\n', log2(7), pass(ok(3)));

% --- 4. the two limits of the noise entropy ------------------------------
% The first is exact and is asserted. The second is only exact for an infinite
% number of trials, so what it really shows is the size of the finite-trial
% bias -- quantified in check 5.
fprintf('4. the two limits of the noise entropy\n');
L   = [2 4 8 12];
one = rand(1, 20000) < p;

S     = spike_entropy(repmat(one, 20, 1), t_bin, L);
ok(4) = max(abs(S.H_noise)) == 0;
fprintf(['   20 identical trials -> noise entropy %.3g bits/s, ' ...
         'information = total   [%s]\n'], max(abs(S.H_noise)), pass(ok(4)));

S = spike_entropy(rand(256, 20000) < p, t_bin, L);
fprintf('   256 independent trials -> true information is 0; what is left\n');
fprintf('   is the finite-trial bias, and it grows with the word length:\n');
fprintf('   L        :%s\n', sprintf('%9d', L));
fprintf('   info     :%s   (total %.0f bits/s)\n\n', ...
        sprintf('%9.1f', S.info), max(S.H_total));

% --- 5. how many trials the noise entropy needs --------------------------
% Descriptive, not pass/fail: there is no threshold here that would not be
% arbitrary. The point is the scaling -- the bias falls roughly as 1/n_trials.
fprintf('5. finite-trial bias: residual information where the true value is 0\n');
bits = rand(512, 20000) < p;
fprintf('   trials |%s\n', sprintf('     L=%-4d', L));
for n = [8 16 32 64 128 256 512]
    S = spike_entropy(bits(1:n,:), t_bin, L);
    fprintf('   %6d |%s\n', n, sprintf('%10.1f', S.info));
end

% --- 6. the shift control sees that floor without knowing the answer ------
fprintf('\n6. the shift control recovers that floor from time-locked data\n');
one = rand(1, 20000) < p;
for n = [16 128]
    % Every trial locked to the same train, plus independent extra spikes.
    b = repmat(one, n, 1) | (rand(n, 20000) < 0.01);
    S = spike_entropy(b, t_bin, L);
    C = spike_entropy(shift_trials(b), t_bin, L);
    fprintf('   n=%-4d info  :%s\n', n, sprintf('%10.1f', S.info));
    fprintf('          floor :%s\n', sprintf('%10.1f', C.info));
end

fprintf('\n%s\n', ternary(all(ok), 'all checks passed', 'SOME CHECKS FAILED'));

% -----------------------------------------------------------------------
function s = pass(tf)
s = ternary(tf, 'ok', 'FAILED');
end

function out = ternary(tf, a, b)
if tf, out = a; else, out = b; end
end
