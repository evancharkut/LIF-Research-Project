function p = lif_params()
% LIF_PARAMS  Default parameters for the leaky integrate-and-fire neuron.
%
%   Single source of truth for the model constants, so that lif_run and the
%   analytic comparisons in lif_fI_curve cannot drift apart.
%
%   Units: R_m in Mohm and I in nA give I*R_m in mV. tau = R_m*C_m with
%   C_m in nF gives tau in ms. So everything below is self-consistent.

p = struct('E_rest',  -65, ...   % resting potential          [mV]
           'V_thresh',-50, ...   % spike threshold            [mV]
           'V_reset', -70, ...   % post-spike reset           [mV]
           'V_spike',  20, ...   % drawn spike height, cosmetic only [mV]
           'R_m',      10, ...   % membrane resistance        [Mohm]
           'C_m',       1, ...   % membrane capacitance       [nF]
           'V_0',     -70, ...   % initial condition          [mV]
           't_ref',     3);      % absolute refractory period [ms]
end
