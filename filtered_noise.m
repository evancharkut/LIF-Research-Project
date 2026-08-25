function [I_e_vect, t_vect] = filtered_noise(t_end, dt, rms_target, dc_offset, f_cut, order)
% FILTERED_NOISE  Band-limited Gaussian noise current for driving the LIF neuron.
%
%   [I_e_vect, t_vect] = filtered_noise(t_end, dt, rms_target, dc_offset, f_cut, order)
%
%   t_end       duration                                     [ms]
%   dt          time step                                    [ms]
%   rms_target  rms of the fluctuating component  (default 1.0)   [nA]
%   dc_offset   constant offset                   (default 1.5)   [nA]
%   f_cut       low-pass cutoff                   (default 40)    [Hz]
%   order       Butterworth order                 (default 4)
%
%   The filter is applied with FILTFILT, so the stimulus is zero-phase and the
%   effective attenuation is that of a filter of twice the stated order.
%   The fluctuating component is scaled *after* filtering, so rms_target is the
%   rms of the delivered stimulus, not of the white noise going in.
%
%   Requires the Signal Processing Toolbox (BUTTER, FILTFILT).

if nargin < 3 || isempty(rms_target), rms_target = 1.0; end
if nargin < 4 || isempty(dc_offset),  dc_offset  = 1.5; end
if nargin < 5 || isempty(f_cut),      f_cut      = 40;  end
if nargin < 6 || isempty(order),      order      = 4;   end

t_vect = dt:dt:t_end;
fs     = 1000 / dt;               % sampling rate [Hz]  (dt is in ms)

if f_cut >= fs/2
    error('filtered_noise:cutoffTooHigh', ...
          'Cutoff (%g Hz) must be below Nyquist (%g Hz).', f_cut, fs/2);
end

% White Gaussian noise. Named I_white rather than "input" so that it does not
% shadow the MATLAB builtin INPUT.
I_white = randn(size(t_vect));

[b, a]  = butter(order, f_cut/(fs/2), 'low');
I_filt  = filtfilt(b, a, I_white);

% Scale the fluctuating part to the requested rms, then add the DC offset.
I_filt   = I_filt - mean(I_filt);
I_filt   = I_filt * (rms_target / std(I_filt));
I_e_vect = I_filt + dc_offset;
end
