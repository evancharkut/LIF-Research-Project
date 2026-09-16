function shifted = shift_trials(bits)
%SHIFT_TRIALS  Circularly shift each trial by a random amount.
%
%   shifted = shift_trials(bits)
%
%   The control for the direct method. The shift destroys the time-locking
%   between trials while leaving each trial's own spike count and word
%   statistics exactly as they were, so the true information in the shifted
%   raster is zero. Whatever SPIKE_ENTROPY still reports for it is the
%   estimator's finite-sample floor at that number of trials and word length.
%
%   Compare the floor with the information before trusting the latter. With
%   too few trials the noise entropy comes out too low, and the shortfall
%   appears as information that is not there.

bits    = bits ~= 0;
shifted = false(size(bits));
n_bins  = size(bits, 2);
for k = 1:size(bits, 1)
    shifted(k, :) = circshift(bits(k, :), randi(n_bins), 2);
end
end
