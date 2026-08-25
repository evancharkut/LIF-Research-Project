% LIF_DC  Response of the leaky integrate-and-fire neuron to a DC current.
% Same structure as the sine-wave script, but with a constant stimulus.

dt       = 0.05;              % [ms]
duration = 1;                 % [s]
t_end    = duration * 1000;   % [ms]
t_vect   = dt:dt:t_end;

I_dc     = 2;                 % [nA]  <-- the "amplitude" of the stimulus
I_e_vect = I_dc * ones(1, numel(t_vect));

[V_vect, spike_times] = lif_run(I_e_vect, dt);

rate = numel(spike_times) / duration;   % [Hz]
fprintf('I = %.2f nA  ->  %d spikes in %.2f s  =  %.1f Hz\n', ...
        I_dc, numel(spike_times), duration, rate);

figure(1); clf
subplot(3,1,1)
plot(t_vect, I_e_vect); ylabel('Current (nA)')
set(gca,'ylim',[0 max(I_e_vect)*1.5])
title(sprintf('DC stimulus, I = %.2f nA  (%.1f Hz)', I_dc, rate))
subplot(3,1,2)
plot(t_vect, V_vect); ylabel('Voltage (mV)')
subplot(3,1,3)
stem(spike_times, spike_times*0+1, 'Marker', 'none'); set(gca,'ylim',[0 1.5])
ylabel('Spikes'); xlabel('Time (ms)')
