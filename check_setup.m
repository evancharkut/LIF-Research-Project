function check_setup()
%CHECK_SETUP  Verify this folder is ready to run the LIF scripts.
%
%   check_setup
%
%   Checks four things that have actually broken this project before:
%     1. base-workspace variables shadowing built-in functions,
%     2. the Signal Processing Toolbox (BUTTER, FILTFILT),
%     3. duplicate copies of the project functions on the path,
%     4. MATLAB new enough for HISTCOUNTS, DISCRETIZE and EXPORTGRAPHICS.

fprintf('\n--- checking setup in %s ---\n\n', pwd);
problems = 0;

% 1. shadowed built-ins ---------------------------------------------------
% A variable named after a function makes every call to it an indexing
% operation. intandfirejit2026 leaves `title` (a 20-character string) in the
% base workspace, which breaks every plotting script until it is cleared.
vars   = evalin('base', 'who');
shadow = {};
for i = 1:numel(vars)
    if ~isempty(which(vars{i})), shadow{end+1} = vars{i}; end %#ok<AGROW>
end
if isempty(shadow)
    fprintf('  OK   no base-workspace variables shadow built-in functions\n');
else
    problems = problems + 1;
    fprintf(2, '  FAIL these variables shadow built-in functions: %s\n', ...
            strjoin(shadow, ', '));
    fprintf(2, '       fix with:  clear %s\n', strjoin(shadow, ' '));
end

% 2. Signal Processing Toolbox -------------------------------------------
missing = {};
for fn = {'butter', 'filtfilt'}
    if isempty(which(fn{1})), missing{end+1} = fn{1}; end %#ok<AGROW>
end
if isempty(missing)
    fprintf('  OK   Signal Processing Toolbox present (butter, filtfilt)\n');
else
    problems = problems + 1;
    fprintf(2, '  FAIL missing: %s\n', strjoin(missing, ', '));
    fprintf(2, ['       filtered_noise needs the Signal Processing Toolbox, ' ...
                'so every\n       noise-driven script will fail without it.\n']);
end

% 3. duplicate project files on the path ---------------------------------
here = pwd;
for fn = {'lif_run', 'lif_params', 'filtered_noise', 'spike_events', ...
          'spike_bits', 'spike_entropy', 'shift_trials'}
    hits = which(fn{1}, '-all');
    if isempty(hits)
        problems = problems + 1;
        fprintf(2, '  FAIL %s not found on the path\n', fn{1});
    elseif ~strcmp(fileparts(hits{1}), here)
        problems = problems + 1;
        fprintf(2, '  FAIL %s resolves to %s\n', fn{1}, hits{1});
        fprintf(2, '       not to the copy in this folder -- cd here first\n');
    elseif numel(hits) > 1
        problems = problems + 1;
        fprintf(2, '  WARN %s has %d copies on the path:\n', fn{1}, numel(hits));
        for h = 1:numel(hits), fprintf(2, '         %s\n', hits{h}); end
    else
        fprintf('  OK   %s resolves to this folder, no duplicates\n', fn{1});
    end
end

% 4. MATLAB version ------------------------------------------------------
for fn = {'histcounts', 'discretize', 'exportgraphics'}
    if isempty(which(fn{1}))
        problems = problems + 1;
        fprintf(2, '  FAIL %s not available -- MATLAB too old (need R2020a+)\n', fn{1});
    end
end
if ~isempty(which('exportgraphics'))
    fprintf('  OK   MATLAB new enough (histcounts, discretize, exportgraphics)\n');
end

fprintf('\n');
if problems == 0
    fprintf('All checks passed. Try:  lif_dc\n\n');
else
    fprintf(2, '%d problem(s) above. Fix them before running the scripts.\n\n', problems);
end
end
