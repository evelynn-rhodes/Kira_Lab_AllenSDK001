function visual_pop_sim_ori_sf

    close all

    %% Simulation parameters
    N_ori = 100;                                  % number of distinct preferred orientations
    N_sf  = 10;                                   % neurons per orientation with different SF tuning
    N     = N_ori * N_sf;                         % total neurons

    % Stimulus dimensions
    stim_oris = 0:10:180;                         % orientations (deg)
    stim_sfs  = logspace(-2, 1, 10);              % spatial freq (cycles/deg), 0.01 to 10

    baseline  = 0.1;                              % baseline dF/F (or spikes/s)
    gain      = 1.0;                              % response gain

    %% Preferred orientations and SFs for each neuron
    % Preferred orientations: uniformly spanning 0..179 deg for N_ori channels
    pref_oris_base = linspace(0, 180, N_ori+1)';  % N_ori × 1
    pref_oris_base = pref_oris_base(1:end-1);

    % Index grid for orientation and SF channel assignments
    [ori_idx, sf_idx] = ndgrid(1:N_ori, 1:N_sf);

    % Flatten to length N = N_ori * N_sf
    pref_oris = pref_oris_base(ori_idx(:));       % N × 1

    % Preferred spatial frequencies (log spaced)
    sf_grid   = logspace(-2, 1, N_sf);            % 0.01 to 10 c/deg
    pref_sfs  = sf_grid(sf_idx(:))';              % N × 1

    %% Tuning parameters
    % Orientation selectivity (von Mises concentration)
    k = 3;
    kappa = k * ones(N,1);

    % Spatial frequency selectivity (Gaussian in log10 space)
    sf_sigma = 0.5;                               % width in log10 units

    %% Convert to radians for orientation tuning
    deg2rad    = pi/180;
    theta_pref = pref_oris * deg2rad;             % N × 1

    %% Build full stimulus set: all orientation × SF combinations
    [ori_grid, sf_grid_idx] = ndgrid(1:numel(stim_oris), 1:numel(stim_sfs));
    stim_ori_list = stim_oris(ori_grid(:));       % n_cond × 1
    stim_sf_list  = stim_sfs(sf_grid_idx(:));     % n_cond × 1
    n_cond        = numel(stim_ori_list);

    %% Compute population responses R: N neurons × n_cond stimuli
    R = zeros(N, n_cond);

    log_pref_sfs = log10(pref_sfs);

    for c = 1:n_cond
        % Orientation tuning
        theta_stim = stim_ori_list(c) * deg2rad;
        dtheta     = theta_stim - theta_pref;     % N × 1
        ori_tuning = exp(kappa .* cos(2*dtheta)); % von Mises-like, 180 deg periodic

        % SF tuning (Gaussian in log10 space)
        log_sf_stim = log10(stim_sf_list(c));
        dlog_sf     = log_sf_stim - log_pref_sfs; % N × 1
        sf_tuning   = exp(-(dlog_sf.^2) ./ (2 * sf_sigma^2));

        % Combined response
        R(:, c) = baseline + gain * (ori_tuning .* sf_tuning);
    end

    % Normalize per neuron so max response ~ baseline + gain
    R = R ./ max(R, [], 2) * (baseline + gain);

    % Add small Gaussian noise (optional)
    noise_sigma = 0;
    % noise_sigma = 0.05;
    R = R + noise_sigma * randn(size(R));
    R = max(R, 0);                                % clip to nonnegative

    fprintf('R size: %d x %d (N x #stim conditions)\n', size(R,1), size(R,2));
    % Here #stim conditions = #oris × #SFs

    %% Reshape R into [N × n_ori × n_sf] for visualization
    n_ori = numel(stim_oris);
    n_sf  = numel(stim_sfs);
    R_3d  = reshape(R, [N, n_ori, n_sf]);

    %% Visualization 0: Example neuron's 2D tuning (SF × orientation)
    % Choose an example neuron index
    neuron_id = 550;  % or any index from 1 to N

    % Extract responses: [n_ori × n_sf], then transpose to [n_sf × n_ori]
    resp_ori_sf = squeeze(R_3d(neuron_id, :, :));   % n_ori × n_sf
    resp_ori_sf = resp_ori_sf';                    % n_sf × n_ori

    figure;
    set(gcf,'OuterPosition',[100 350 700 700])

    imagesc(stim_oris, 1:n_sf, resp_ori_sf);
    set(gca, 'YDir', 'normal');  % low SF at bottom, high at top
    colormap(parula);
    colorbar;

    xlabel('Orientation (deg)');
    ylabel('Spatial frequency (cycles/deg)');
    title(sprintf('Neuron %d: orientation × SF tuning', neuron_id), 'FontSize', 20);

    % Label y-axis with actual SF values
    yticks(1:n_sf);
    yticklabels(arrayfun(@(x) sprintf('%.3g', x), stim_sfs, 'UniformOutput', false));


    %% Visualization 1: Population responses for a single SF slice
    % Choose an SF index to visualize (for example, middle SF)
    sf_idx_vis = ceil(n_sf / 2);

    figure;
    set(gcf,'OuterPosition',[700 350 700 700])
    imagesc(stim_oris, 1:N, squeeze(R_3d(:, :, sf_idx_vis)));
    xlabel('Stimulus orientation (deg)');
    ylabel('Neuron index');
    title(sprintf('Population responses at SF = %.3f c/deg', stim_sfs(sf_idx_vis)), 'FontSize', 20);
    colormap(parula);
    colorbar;

    %% Visualization 2: Example tuning of a few neurons across orientation at a fixed SF
    figure('Color','w'); hold on
    set(gcf,'OuterPosition',[700 350 700 700])

    % Pick first 5 neurons for illustration
    example_neurons = 1:5;
    plot(stim_oris, squeeze(R_3d(example_neurons, :, sf_idx_vis))', ...
        '-o', 'LineWidth', 1.2);

    xlabel('Orientation (deg)');
    ylabel('Response');
    title(sprintf('Example orientation tuning (first 5 neurons, SF = %.3f c/deg)', ...
          stim_sfs(sf_idx_vis)), 'FontSize', 20);
    grid on

    %% Visualization 3: Example SF tuning for a single preferred orientation
    % Choose one orientation channel and plot SF tuning of a few neurons with that pref_ori
    ori_idx_example = round(N_ori / 2);                      % mid orientation channel
    neurons_in_ori  = find(pref_oris == pref_oris_base(ori_idx_example));

    % Take up to 5 neurons from this orientation, if available
    neurons_in_ori  = neurons_in_ori(1:min(5,numel(neurons_in_ori)));

    % Fix stimulus orientation to the corresponding orientation
    [~, stim_ori_idx_example] = min(abs(stim_oris - pref_oris_base(ori_idx_example)));

    figure('Color','w'); hold on
    set(gcf,'OuterPosition',[700 350 700 700])

    for ni = 1:numel(neurons_in_ori)
        n_id = neurons_in_ori(ni);
        % Responses across SFs at fixed stimulus orientation
        resp_sf = squeeze(R_3d(n_id, stim_ori_idx_example, :));  % n_sf × 1
        semilogx(stim_sfs, resp_sf, '-o', 'LineWidth', 1.2);
    end

    xlabel('Spatial frequency (cycles/deg)');
    ylabel('Response');
    title(sprintf('Example SF tuning (neurons with pref ori ≈ %.1f°)', ...
          pref_oris_base(ori_idx_example)), 'FontSize', 20);
    grid on

    %% PCA on population responses across all ori × SF conditions
    % Each stimulus condition is one observation, neurons are features
    X = R.';                          % [n_cond × N]
    X(isnan(X)) = 0;
    X = X - mean(X,1);

    [coeff, score, latent] = pca(X);

    %% Cyclic colormap for orientations (ignoring SF in color)
    nOri_color = numel(stim_oris);
    angles = linspace(0, 2*pi, nOri_color+1);
    angles(end) = [];                 % drop duplicate
    hsv_colors = hsv2rgb([angles'/(2*pi), ones(nOri_color,1), ones(nOri_color,1)]);
    cmap = hsv_colors;

    % For each condition, map its orientation to a color index
    [~, ori_idx_for_cond] = ismember(stim_ori_list, stim_oris);

    %% Visualization 4: Top 3 PCs, color coded by orientation
    figure('Color','w'); hold on
    set(gcf,'OuterPosition',[700 350 700 700])

    % Connect points in stimulus order with black line
    plot3(score(:,1), score(:,2), score(:,3), '-k', 'LineWidth', 1.2);
    plot3([score(end,1) score(1,1)], ...
          [score(end,2) score(1,2)], ...
          [score(end,3) score(1,3)], ...
          '-k', 'LineWidth', 1.2);

    % Scatter each stimulus condition with orientation based color
    for i = 1:n_cond
        this_color = cmap(ori_idx_for_cond(i), :);
        scatter3(score(i,1), score(i,2), score(i,3), ...
            60, this_color, 'filled', 'MarkerEdgeColor','k');
    end

    xlabel('PC1');
    ylabel('PC2');
    zlabel('PC3');

    explVar = 100 * latent / sum(latent);
    title(sprintf('Population PCA (%.1f%% / %.1f%% / %.1f%%)', ...
        explVar(1), explVar(2), explVar(3)), 'FontSize', 20);

    colormap(cmap);
    c = colorbar;

    % Orientation ticks on colorbar
    tick_oris = 0:30:180;
    tick_positions = tick_oris / max(stim_oris);
    c.Ticks = tick_positions;
    c.TickLabels = string(tick_oris);
    c.Label.String = 'Orientation (deg)';

    grid on;
    axis equal;
    view(3);
    add_margin(score);

end

function add_margin(score)
    % Compute axis limits with 10 percent margins
    margin_ratio = 0.10;

    x_min = min(score(:,1)); x_max = max(score(:,1));
    y_min = min(score(:,2)); y_max = max(score(:,2));
    z_min = min(score(:,3)); z_max = max(score(:,3));

    x_range = x_max - x_min;
    y_range = y_max - y_min;
    z_range = z_max - z_min;

    xlim([x_min - margin_ratio*x_range, x_max + margin_ratio*x_range]);
    ylim([y_min - margin_ratio*y_range, y_max + margin_ratio*y_range]);
    zlim([z_min - margin_ratio*z_range, z_max + margin_ratio*z_range]);
end
