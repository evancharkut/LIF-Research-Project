% LIF_FI_CURVE  Firing rate as a function of DC stimulus amplitude (f-I curve).

dt       = 0.05;              % [ms]
duration = 2;                 % [s]  (longer run = smoother rate estimate)
t_end    = duration * 1000;   % [ms]
t_vect   = dt:dt:t_end;

I_list = 0:0.05:10;           % [nA] stimulus amplitudes to sweep
rates  = zeros(size(I_list));

for k = 1:numel(I_list)
    I_e_vect  = I_list(k) * ones(1, numel(t_vect));
    [~, spike_times] = lif_run(I_e_vect, dt);
    rates(k)  = numel(spike_times) / duration;      % [Hz]
end

% --- analytic prediction, for comparison -------------------------------
% Between spikes V(t) relaxes toward I*R_m + E_rest starting from V_reset.
% Time to reach V_thresh:  ISI = tau*ln( (I*R+E_rest-V_reset)/(I*R+E_rest-V_thresh) )
p = lif_params();             % same constants the simulation used
tau       = p.R_m*p.C_m;
V_inf     = I_list*p.R_m + p.E_rest;
I_rheo    = (p.V_thresh - p.E_rest)/p.R_m;          % 1.5 nA here
theory    = zeros(size(I_list));
fire      = V_inf > p.V_thresh;
ISI       = tau*log( (V_inf(fire)-p.V_reset) ./ (V_inf(fire)-p.V_thresh) );
theory(fire) = 1000 ./ (ISI + p.t_ref);             % [Hz]

figure(2); clf
plot(I_list, rates, 'o', 'MarkerSize', 3); hold on
plot(I_list, theory, 'r-', 'LineWidth', 1.2)
xline(I_rheo, 'k--');
xlabel('DC stimulus amplitude I (nA)')
ylabel('Firing rate (Hz)')
legend('simulation', 'analytic', 'rheobase', 'Location', 'southeast')
title('Leaky integrate-and-fire: firing rate vs. DC amplitude')
grid on

above = I_list > I_rheo;
fprintf('Rheobase (minimum current to fire) = %.2f nA\n', I_rheo);
fprintf('Max rate as I -> inf = %.1f Hz (set by t_ref = %g ms)\n', ...
        1000/p.t_ref, p.t_ref);
fprintf('Max |simulation - analytic| above rheobase = %.2f Hz\n', ...
        max(abs(rates(above) - theory(above))));
