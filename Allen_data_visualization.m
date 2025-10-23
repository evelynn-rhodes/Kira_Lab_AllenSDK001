function Allen_data_visualization

    close all

    % Load dataset
    file_name = '/Users/shin/Documents/GitHub/AllenSDK/data/allen_dataset.mat';
    data = load(file_name);

    metadata   = struct_to_table(data.cell_metadata);
    stim_table = struct_to_table(data.stim_table);

    dff_traces = data.dff_traces_sorted;
    timestamps = data.timestamps(:);

    % Remove blank rows (NaN stimuli) from stim_table
    valid_idx = ~isnan(stim_table.orientation);
    stim_valid = stim_table(valid_idx, :);

    % Extract unique stimulus values
    orientations = unique(stim_valid.orientation);
    sfs          = unique(stim_valid.spatial_frequency);
    phases       = unique(stim_valid.phase);

    nOri  = numel(orientations);
    nSF   = numel(sfs);
    nPh   = numel(phases);
    nCell = size(dff_traces, 1);

    % Compute mean response per stimulus type as before
    [stim_types, ~, stim_idx] = unique( ...
        [stim_valid.orientation, stim_valid.spatial_frequency, stim_valid.phase], ...
        'rows');

    nStim = size(stim_types,1);
    mean_response = nan(nCell, nStim);

    for s = 1:nStim
        trials = find(stim_idx == s);
        resp_all = [];
        for t = trials'
            frame_start = stim_valid.start(t);
            frame_end   = stim_valid.end(t);
            if frame_end > size(dff_traces, 2)
                continue
            end
            resp = mean(dff_traces(:, frame_start:frame_end), 2, 'omitnan');
            resp_all = [resp_all, resp];
        end
        mean_response(:, s) = mean(resp_all, 2, 'omitnan');
    end

    % --- Convert mean_response into a 4-D tuning matrix ---
    resp4D = nan(nOri, nSF, nPh, nCell);

    for iOri = 1:nOri
        for iSF = 1:nSF
            for iPh = 1:nPh
                % find the corresponding stimulus index
                match = (stim_types(:,1) == orientations(iOri)) & ...
                        (stim_types(:,2) == sfs(iSF)) & ...
                        (stim_types(:,3) == phases(iPh));
                if any(match)
                    resp4D(iOri, iSF, iPh, :) = mean_response(:, match);
                end
            end
        end
    end

    resp3D = squeeze(mean(resp4D,3));


        % --- Run PCA on mean responses (averaged over phase) ---
    % resp3D: [nOri x nSF x nCell]
    [nOri, nSF, nCell] = size(resp3D);

    % Flatten cell responses, so each stimulus condition = one row
    X = reshape(resp3D, nOri*nSF, nCell);   % [stimuli x cells]

    % Replace NaNs with zeros
    X(isnan(X)) = 0;

    % Run PCA (stimuli as observations)
    [coeff, score, latent] = pca(X);

    % score: coordinates of each stimulus condition in PC space
    % Build label vectors for plotting
    [ori_grid, sf_grid] = ndgrid(1:nOri, 1:nSF);
    ori_idx = ori_grid(:);
    sf_idx  = sf_grid(:);

    % Map orientation/SF indices to actual values
    ori_vals = orientations(ori_idx);
    sf_vals  = sfs(sf_idx);

    % Define unique markers for SFs
    markers = {'o', 's', '^', 'd', 'v', '>'};
    nMarkers = numel(markers);
    

    % --- Plot each SF separately ---
    cmap = parula(nOri);
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h'};
    nMarkers = numel(markers);

    for iSF = 1:nSF
        sf_mask = (sf_idx == iSF);
        pts = score(sf_mask, 1:3);
        ori_subidx = ori_idx(sf_mask);

        % Sort by orientation for smooth connection
        [~, sortIdx] = sort(ori_subidx);
        pts_sorted = pts(sortIdx, :);
        ori_sorted = ori_subidx(sortIdx);

        % Create new figure for each SF
        figure('Name', sprintf('SF = %.3f', sfs(iSF)), 'Color', 'w');
        hold on;

        % Plot all orientation points (colored)
        for k = 1:numel(ori_sorted)
            scatter3(pts_sorted(k,1), pts_sorted(k,2), pts_sorted(k,3), ...
                60, cmap(ori_sorted(k),:), markers{mod(iSF-1,nMarkers)+1}, 'filled');
        end

        % Connect all orientations in order
        plot3(pts_sorted(:,1), pts_sorted(:,2), pts_sorted(:,3), ...
            '-k', 'LineWidth', 1.2);

        % --- Close the loop by connecting last → first point ---
        plot3([pts_sorted(end,1) pts_sorted(1,1)], ...
              [pts_sorted(end,2) pts_sorted(1,2)], ...
              [pts_sorted(end,3) pts_sorted(1,3)], ...
              '-k', 'LineWidth', 1.2);

        % Axes and title
        xlabel('PC1');
        ylabel('PC2');
        zlabel('PC3');
        title(sprintf('PCA projection — SF = %.3f', sfs(iSF)));

        % Formatting
        grid on;
        axis equal;
        set(gca, 'FontSize', 12);

        % Colorbar for orientations
        colormap(cmap);
        c = colorbar;
        c.Ticks = linspace(0,1,nOri);
        c.TickLabels = string(orientations);
        c.Label.String = 'Orientation (deg)';
        c.Label.FontSize = 12;

        view(3);  % 3D view
    end

    % Print explained variance
    explVar = 100 * latent / sum(latent);
    fprintf('Explained variance (%%): PC1=%.1f, PC2=%.1f, PC3=%.1f\n', ...
        explVar(1), explVar(2), explVar(3));




    % --- Example: visualize one cell’s tuning map (orientation × SF) at one phase ---
    cell_idx = 1;
    ph_idx   = 1;

    figure;
    imagesc(1:5, orientations, squeeze(resp3D(:,:,cell_idx)));

    % Axis labels and ticks
    xlabel('Spatial frequency (cycles/deg)');
    ylabel('Orientation (deg)');
    title(sprintf('Cell %d, phase = %.2f', cell_idx, phases(ph_idx)));
    colorbar;
    clim([0 0.05]);

    % Set tick positions and labels to match actual values
    xticks(1:5);
    yticks(orientations);
    xticklabels(string(sfs));
    yticklabels(string(orientations));

    axis tight;
    set(gca,'YDir','normal');  % keep low→high orientation bottom→top

end

function T = struct_to_table(s)
    fn = fieldnames(s);

    % --- Infer number of rows N from any field ---
    N = [];
    for k = 1:numel(fn)
        v = s.(fn{k});
        if ischar(v)
            N = size(v,1); break
        elseif isvector(v)
            N = numel(v); break
        elseif iscell(v)
            sz = size(v);
            N = max(sz(1), sz(2)); break
        end
    end
    if isempty(N)
        error('Could not infer row count from struct fields.');
    end

    % --- Build table with normalized columns (N×1) ---
    T = table();
    for k = 1:numel(fn)
        name = fn{k};
        v = s.(name);

        if ischar(v)
            v = string(cellstr(v));
        end
        if iscell(v) || isa(v,'string')
            if isrow(v), v = v(:); end
        end
        if isnumeric(v) || islogical(v)
            if isvector(v), v = v(:); end
        end
        nv = size(v,1);
        if nv ~= N
            if isscalar(v)
                v = repmat(v, N, 1);
            else
                error('Field "%s" has %d rows; expected %d.', name, nv, N);
            end
        end
        T.(name) = v;
    end
end
