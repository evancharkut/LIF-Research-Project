function make_figures()
% MAKE_FIGURES  Run each demo script and save its figure into figures/.
% Run once from the repository root to regenerate the images in README.md.

if ~exist('figures', 'dir'), mkdir('figures'); end

scripts = {'lif_dc', 'lif_fI_curve', 'lif_filtered_noise'};
fignums = [1 2 3];
names   = {'dc_step', 'fI_curve', 'frozen_noise_raster'};

for idx = 1:numel(scripts)
    export_one(scripts{idx}, fignums(idx), ...
               fullfile('figures', [names{idx} '.png']));
    fprintf('saved figures/%s.png\n', names{idx});
end
end

% -----------------------------------------------------------------------
function export_one(script_name, fig_num, out_path)
% RUN executes a script in the *caller's* workspace, and the demo scripts
% use `k` as a loop counter. Calling it from here confines the damage to
% this throwaway workspace instead of clobbering the loop index above.
run(script_name);
exportgraphics(figure(fig_num), out_path, 'Resolution', 150);
end
