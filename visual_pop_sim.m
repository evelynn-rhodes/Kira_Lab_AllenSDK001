function visual_pop_sim

    close all

    % Simulation parameters
    N = 300;                                      % number of neurons
    % stim_oris = [0 30 60 90 120 150];             % degrees (length = 6)
    stim_oris = 0:179.9;                            % degrees (length = 6)
    baseline = 0.1;                               % baseline dF/F (or spikes/s)
    gain     = 1.0;                               % response gain
    
    % Preferred orientations (uniformly spanning 0..179 deg)
    pref_oris = linspace(0, 179.9, N)';             % N×1
    
    % Selectivity: concentration (kappa) per neuron (larger = sharper tuning)
    % You can adjust the range to control selectivity spread.
    k = 3;
    kappa = k * ones(N,1);
    % kappa = linspace(0.5, 8, N)';                 % N×1
    
    % Convert degrees to radians for von Mises
    deg2rad = pi/180;
    theta_stim = stim_oris * deg2rad;             % 1×S
    theta_pref = pref_oris * deg2rad;             % N×1
    
    % Von Mises-like orientation tuning (periodic with 180° => use 2*angle)
    % Response R = baseline + gain * exp(kappa * cos(2*(θ - θ_pref))) - exp(kappa)
    % The "- exp(kappa)" term centers the peak at gain above baseline (optional).
    % To keep it simple and positive, we’ll omit centering and rescale.
    R = zeros(N, numel(stim_oris));
    for s = 1:numel(stim_oris)
        dtheta = theta_stim(s) - theta_pref;      % N×1
        R(:, s) = baseline + gain * exp(kappa .* cos(2*dtheta));
    end
    
    % Normalize per neuron so max response ~ baseline + gain
    R = R ./ max(R, [], 2) * (baseline + gain);
    
    % Add small Gaussian noise (optional)
    noise_sigma = 0;
    % noise_sigma = 0.05;
    R = R + noise_sigma * randn(size(R));
    R = max(R, 0);                                % clip to nonnegative
    
    % Outputs:
    % R          -> N × 6 response matrix
    % pref_oris  -> N × 1 preferred orientation (deg)
    % kappa      -> N × 1 selectivity parameter
    
    % Quick check
    fprintf('R size: %d x %d (N x #stim)\n', size(R,1), size(R,2));

    figure;
    imagesc(R)
    
    % Example visualization: a few neurons’ tuning across the 6 orientations
    figure('Color','w'); hold on
    plot(stim_oris, R(1:5,:)', '-o', 'LineWidth', 1.2);
    xlabel('Orientation (deg)'); ylabel('Response');
    title('Example simulated tuning (first 5 neurons)');
    grid on

        % --- PCA on population responses across stimuli ---
    % Each stimulus orientation = observation, neurons = features
    X = R.';                       % [#stim x N]
    X(isnan(X)) = 0;
    X = X - mean(X,1);

    [coeff, score, latent] = pca(X);

    % --- Visualization: Top 3 PCs, color-coded by orientation ---
    % --- Cyclic colormap for orientations ---
    nOri = numel(stim_oris);
    angles = linspace(0, 2*pi, nOri+1);   % one full cycle
    angles(end) = [];                     % drop last to avoid duplicate color
    hsv_colors = hsv2rgb([angles'/(2*pi), ones(nOri,1), ones(nOri,1)]);
    cmap = hsv_colors;  % cyclic colormap (wraps smoothly)


    figure('Color','w'); hold on

    % Connect points (in stimulus order) with black line
    plot3(score(:,1), score(:,2), score(:,3), '-k', 'LineWidth', 1.2);
    plot3([score(end,1) score(1,1)], ...
          [score(end,2) score(1,2)], ...
          [score(end,3) score(1,3)], ...
          '-k', 'LineWidth', 1.2);  % close the loop

    % Scatter each stimulus point with orientation-based color
    for i = 1:numel(stim_oris)
        scatter3(score(i,1), score(i,2), score(i,3), ...
            100, cmap(i,:), 'filled', 'MarkerEdgeColor','k');
    end

    % Label each point with orientation (optional)
    if 0
        for i = 1:numel(stim_oris)
            text(score(i,1), score(i,2), score(i,3), sprintf('  %d°', stim_oris(i)), ...
                'FontSize', 10, 'Color', [0 0 0]);
        end
    end

    xlabel('PC1');
    ylabel('PC2');
    zlabel('PC3');

    % Show explained variance in title
    explVar = 100 * latent / sum(latent);
    title(sprintf('Population PCA (%.1f%% / %.1f%% / %.1f%%)', ...
        explVar(1), explVar(2), explVar(3)));

    colormap(cmap);
    c = colorbar;
    c.Ticks = linspace(0,1,numel(stim_oris));
    c.TickLabels = string(stim_oris);
    c.Label.String = 'Orientation (deg)';
    c.Label.FontSize = 12;

    grid on;
    axis equal;
    view(3);


end
