function [V_vect, spike_times, t_vect] = lif_run(I_e_vect, dt, p)
% LIF_RUN  Leaky integrate-and-fire neuron driven by an arbitrary current.
%
%   [V_vect, spike_times, t_vect] = lif_run(I_e_vect, dt, p)
%
%   I_e_vect  injected current, one value per time step   [nA]
%   dt        time step                                   [ms]
%   p         (optional) struct of parameters; any field you leave out
%             falls back to the default in LIF_PARAMS.
%
%   V_vect      membrane potential, one value per time step  [mV]. Spikes are
%               drawn at p.V_spike for display only; the integration state is
%               a separate scalar, so the drawn spike never feeds back into
%               the dynamics.
%   spike_times spike times                                  [ms]
%   t_vect      time base, (1:N)*dt                          [ms]
%
%   Units: R_m in Mohm and I in nA give I*R_m in mV. tau = R_m*C_m with
%   C_m in nF gives tau in ms. So everything below is self-consistent.
%
%   The update is exponential Euler, which is exact for a current that is
%   constant across the step, so the result is insensitive to dt well before
%   forward Euler would have converged.

if nargin < 3, p = struct(); end
d = lif_params();
f = fieldnames(d);
for k = 1:numel(f)
    if ~isfield(p, f{k}), p.(f{k}) = d.(f{k}); end
end
tau = p.R_m * p.C_m;                 % [ms]

N       = numel(I_e_vect);
t_vect  = (1:N) * dt;
V_vect  = zeros(1, N);
V_vect(1) = p.V_0;
V       = p.V_0;

% Preallocate to the largest number of spikes the refractory period allows.
spike_times = zeros(1, ceil(N*dt / p.t_ref) + 1);
n_spikes    = 0;
ref_count   = 0;

for i = 2:N
    if ref_count > 0
        % --- absolute refractory period: clamp at reset, ignore input ---
        V = p.V_reset;
        V_vect(i) = V;
        ref_count = ref_count - 1;
    else
        % --- exponential Euler step toward the steady state I*R + E_rest ---
        V_inf = I_e_vect(i)*p.R_m + p.E_rest;
        V     = V_inf + (V - V_inf)*exp(-dt/tau);

        if V > p.V_thresh
            V_vect(i)             = p.V_spike;   % draw the spike at THIS index
            n_spikes              = n_spikes + 1;
            spike_times(n_spikes) = t_vect(i);
            V                     = p.V_reset;
            ref_count             = round(p.t_ref/dt);
        else
            V_vect(i) = V;
        end
    end
end

spike_times = spike_times(1:n_spikes);
end
