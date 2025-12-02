function visual_pop_sim_ori_sf_irregular

    close all

    %% =========================================
    %  FLAGS: irregularity switches
    %% =========================================
    FLAG_HET_KAPPA          = 0;  % neuron specific orientation selectivity
    FLAG_HET_SF_SIGMA       = 0;  % neuron specific SF bandwidth
    FLAG_ORI_DEP_SF_GAIN    = 1;  % SF gain depends on preferred orientation
    FLAG_BIASED_PREF_SF     = 0;  % non uniform preferred SF distribution
    FLAG_ORI_DEP_PREF_SF    = 0;  % preferred SF depends on preferred orientation
    FLAG_HET_BASE_GAIN      = 0;  % neuron specific baseline and gain
    FLAG_NORMALIZE_PER_NEUR = 0;  % per neuron normalization

    %% Simulation parameters
    N_ori = 100;                                  % number of distinct preferred orientations
    N_sf  = 10;                                   % neurons per orientation with different SF tuning
    N     = N_ori * N_sf;                         % total neurons

    % Stimulus dimensions
    stim_oris = 0:10:180;                         % orientations (deg)
    stim_sfs  = logspace(-3, 2, 15);              % spatial freq (cycles/deg), 0.001 to 100

    baseline_global  = 0.1;                       % global baseline (used if FLAG_HET_BASE_GAIN==0)
    gain_global      = 1.0;                       % global response gain

    %% Preferred orientations and SFs for each neuron
    % Preferred orientations: uniformly spanning 0..179 deg for N_ori channels
    pref_oris_base = linspace(0, 180, N_ori+1)';  % N_ori × 1
    pref_oris_base = pref_oris_base(1:end-1);

    % Index grid for orientation and SF channel assignments
    [ori_idx, sf_idx] = ndgrid(1:N_ori, 1:N_sf);

    % Flatten to length N = N_ori * N_sf
    pref_oris = pref_oris_base(ori_idx(:));       % N × 1

    % Preferred spatial frequencies (log spaced)
    if FLAG_BIASED_PREF_SF
        % Slight bias toward lower SFs
        sf_grid   = logspace(-2, 0.7, N_sf);      % 0.01 to about 5 c/deg
    else
        sf_grid   = logspace(-2, 1, N_sf);        % 0.01 to 10 c/deg
    end
    pref_sfs  = sf_grid(sf_idx(:))';              % N × 1

    %% Tuning parameters

    % Orientation selectivity (von Mises concentration)
    if FLAG_HET_KAPPA
        kappa_mean = 3;
        kappa_sd   = 1;
        kappa = kappa_mean + kappa_sd * randn(N,1);
        kappa = max(0.2, kappa);
    else
        kappa = 3 * ones(N,1);
    end

    % Spatial frequency selectivity (Gaussian in log10 space)
    if FLAG_HET_SF_SIGMA
        sf_sigma_mean = 0.5;
        sf_sigma_sd   = 0.2;
        sf_sigma = sf_sigma_mean + sf_sigma_sd * randn(N,1);
        sf_sigma = max(0.1, sf_sigma);
    else
        sf_sigma = 0.5 * ones(N,1);               % width in log10 units
    end

    %% Convert to radians for orientation tuning
    deg2rad    = pi/180;
    theta_pref = pref_oris * deg2rad;             % N × 1

    %% Preferred SF in log space, with optional orientation dependence
    log_pref_sfs = log10(pref_sfs);

    if FLAG_ORI_DEP_PREF_SF
        % Example: preferred SF slightly higher for oblique orientations
        log_pref_sfs = log_pref_sfs + 0.3 * cos(2 * theta_pref);
        pref_sfs     = 10 .^ log_pref_sfs;
    end

    % Recompute in case it changed
    log_pref_sfs = log10(pref_sfs);

    %% Orientation dependent SF gain
    sf_gain_neuron = ones(N,1);
    if FLAG_ORI_DEP_SF_GAIN
        % Example: neurons preferring cardinals have stronger SF gain
        cardinal_mod   = 1 + 0.42 * (cos(2 * theta_pref)).^2; % 1 to 1.5
        sf_gain_neuron = cardinal_mod;
    end

    %% Neuron specific baseline and gain
    if FLAG_HET_BASE_GAIN
        baseline_neuron = 0.05 + 0.05 * rand(N,1);   % 0.05 to 0.10
        gain_neuron     = 0.8  + 0.6  * rand(N,1);   % 0.8 to 1.4
    else
        baseline_neuron = baseline_global * ones(N,1);
        gain_neuron     = gain_global * ones(N,1);
    end

    %% Build full stimulus set: all orientation × SF combinations
    [ori_grid, sf_grid_idx] = ndgrid(1:numel(stim_oris), 1:numel(stim_sfs));
    stim_ori_list = stim_oris(ori_grid(:));       % n_cond × 1
    stim_sf_list  = stim_sfs(sf_grid_idx(:));     % n_cond × 1
    n_cond        = numel(stim_ori_list);

    %% Compute population responses R: N neurons × n_cond stimuli
    R = zeros(N, n_cond);

    for c = 1:n_cond
        % Orientation tuning
        theta_stim = stim_ori_list(c) * deg2rad;
        dtheta     = theta_stim - theta_pref;         % N × 1
        ori_tuning = exp(kappa .* cos(2*dtheta));     % von Mises like, 180 deg periodic

        % SF tuning (Gaussian in log10 space)
        log_sf_stim = log10(stim_sf_list(c));
        dlog_sf     = log_sf_stim - log_pref_sfs;     % N × 1
        sf_tuning   = exp(-(dlog_sf.^2) ./ (2 * sf_sigma.^2));

        % Apply orientation dependent SF gain if enabled
        sf_tuning = sf_gain_neuron .* sf_tuning;

        % Combined response with neuron specific baseline and gain
        R(:, c) = baseline_neuron + gain_neuron .* (ori_tuning .* sf_tuning);
    end

    % Optional per neuron normalization
    if FLAG_NORMALIZE_PER_NEUR
        R = R ./ max(R, [], 2);
        R(isnan(R)) = 0;
        R = R * (baseline_global + gain_global);
    end

    % Add small Gaussian noise (optional)
    noise_sigma = 0;
    % noise_sigma = 0.05;
    if noise_sigma > 0
        R = R + noise_sigma * randn(size(R));
        R = max(R, 0);                              % clip to nonnegative
    end

    fprintf('R size: %d x %d (N x #stim conditions)\n', size(R,1), size(R,2));
    % Here #stim conditions = #oris × #SFs

    %% Reshape R into [N × n_ori × n_sf] for visualization
    n_ori = numel(stim_oris);
    n_sf  = numel(stim_sfs);
    R_3d  = reshape(R, [N, n_ori, n_sf]);

    %% Visualization 0: Example neuron's 2D tuning (SF × orientation)
    neuron_id = 100;                        % example neuron index
    neuron_id = min(neuron_id, N);          % safety

    % Extract responses: [n_ori × n_sf], then transpose to [n_sf × n_ori]
    resp_ori_sf = squeeze(R_3d(neuron_id, :, :));   % n_ori × n_sf
    resp_ori_sf = resp_ori_sf';                     % n_sf × n_ori

    figure;
    set(gcf,'OuterPosition',[100 350 700 700])

    imagesc(stim_oris, 1:n_sf, resp_ori_sf);
    set(gca, 'YDir', 'normal');  % low SF at bottom, high at top
    colormap(parula);
    colorbar;

    xlabel('Orientation (deg)');
    ylabel('Spatial frequency (cycles/deg)');
    title(sprintf('Neuron %d: orientation × SF tuning', neuron_id), 'FontSize', 20);

    % Label y axis with actual SF values
    yticks(1:n_sf);
    yticklabels(arrayfun(@(x) sprintf('%.3g', x), stim_sfs, 'UniformOutput', false));

    %% Visualization 1: Population responses for a single SF slice
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

    example_neurons = 1:min(5,N);
    plot(stim_oris, squeeze(R_3d(example_neurons, :, sf_idx_vis))', ...
        '-o', 'LineWidth', 1.2);

    xlabel('Orientation (deg)');
    ylabel('Response');
    title(sprintf('Example orientation tuning (first 5 neurons, SF = %.3f c/deg)', ...
          stim_sfs(sf_idx_vis)), 'FontSize', 20);
    grid on

    %% Visualization 3: Example SF tuning for a single preferred orientation
    ori_idx_example = round(N_ori / 2);                      % mid orientation channel
    neurons_in_ori  = find(pref_oris == pref_oris_base(ori_idx_example));

    neurons_in_ori  = neurons_in_ori(1:min(5,numel(neurons_in_ori)));

    [~, stim_ori_idx_example] = min(abs(stim_oris - pref_oris_base(ori_idx_example)));

    figure('Color','w'); hold on
    set(gcf,'OuterPosition',[700 350 700 700])

    for ni = 1:numel(neurons_in_ori)
        n_id = neurons_in_ori(ni);
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

    %% Cumulative explained variance plot
    explVar    = 100 * latent / sum(latent);
    cumExplVar = cumsum(explVar);

    figure('Color','w');
    set(gcf,'OuterPosition',[300 300 600 500]);

    subplot(2,1,1)
    numPCs = numel(explVar);
    plot(1:numPCs, cumExplVar, '-o', 'LineWidth', 1.5);
    xlabel('Number of principal components');
    ylabel('Cumulative explained variance (%)');
    title('Cumulative variance explained by PCs', 'FontSize', 16);
    grid on;

    subplot(2,1,2)
    plot(1:numPCs, explVar, '-o', 'LineWidth', 1.5);
    xlabel('Principal component');
    ylabel('Variance explained (%)');
    grid on;

    %% Color coding: orientation = hue, spatial frequency = brightness
    % Base hues for each orientation (for colorbar)
    nOri_color = numel(stim_oris);
    angles = linspace(0, 2*pi, nOri_color+1);
    angles(end) = [];                                 % drop duplicate
    hues = angles'/(2*pi);                            % nOri_color × 1

    % For each condition, map its orientation to an index
    [~, ori_idx_for_cond] = ismember(stim_ori_list, stim_oris);

    % Normalize spatial frequency to brightness proxy (here used for saturation)
    sf_min = min(stim_sf_list);
    sf_max = max(stim_sf_list);
    sf_norm = (stim_sf_list - sf_min) / (sf_max - sf_min);   % 0 to 1
    brightness = 0.1 + 0.9 * sf_norm;                        % avoid very desaturated points

    % Build RGB colors for each condition
    colors_sf = zeros(n_cond, 3);
    for i = 1:n_cond
        hue = hues(ori_idx_for_cond(i));   % orientation
        sat = brightness(i);               % spatial frequency encoded as saturation
        val = 1;
        colors_sf(i, :) = hsv2rgb([hue, sat, val]);
    end

    %% Visualization 4: Top 3 PCs, color coded by ori (hue) and SF (brightness)
    figure('Color','w'); hold on
    set(gcf,'OuterPosition',[700 350 700 700])

    % Connect points in stimulus order with black line
    plot3(score(:,1), score(:,2), score(:,3), '-k', 'LineWidth', 1.2);
    plot3([score(end,1) score(1,1)], ...
          [score(end,2) score(1,2)], ...
          [score(end,3) score(1,3)], ...
          '-k', 'LineWidth', 1.2);

    % Scatter each stimulus condition with combined color
    for i = 1:n_cond
        scatter3(score(i,1), score(i,2), score(i,3), ...
            60, colors_sf(i,:), 'filled', 'MarkerEdgeColor','k');
    end

    xlabel('PC1');
    ylabel('PC2');
    zlabel('PC3');

    title(sprintf('Population PCA (%.1f%% / %.1f%% / %.1f%%)', ...
        explVar(1), explVar(2), explVar(3)), 'FontSize', 20);

    % Colorbar that shows orientation hues at maximum saturation
    cmap = hsv2rgb([hues, ones(nOri_color,1), ones(nOri_color,1)]);
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
