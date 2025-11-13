%% ============================================================
% Allen Static Gratings Full Analysis (Single Cell + 3D Cube)
% ============================================================

clear; clc; close all

%% -------- Set Experiment Folder ----------
exp_id = 500964514;   % <-- change for other experiments
exp_dir = fullfile('.', sprintf('exp_%d', exp_id));

stim_csv = fullfile(exp_dir, sprintf('static_gratings_stim_table_%d.csv', exp_id));
ts_csv   = fullfile(exp_dir, sprintf('timestamps_%d.csv', exp_id));
dff_csv  = fullfile(exp_dir, sprintf('dff_traces_%d.csv', exp_id));

%% -------- Load Data ----------
stim = readtable(stim_csv,'VariableNamingRule','preserve');
ts   = readtable(ts_csv);
dff  = readtable(dff_csv);

time_s   = ts.time_s;
cell_ids = dff{:,1};
F        = dff{:,2:end};

dt = median(diff(time_s));
[nCell, T_f] = size(F);
fprintf('Loaded: %d cells, %d timepoints, dt=%.4f\n', nCell, T_f, dt);

%% -------- Extract unique conditions ----------
oris = unique(stim.orientation);
sfs  = unique(stim.spatial_frequency);
phs  = unique(stim.phase);

% Expected bins (Allen static gratings)
ori_expected = [0 30 60 90 120 150];
sf_expected  = [0.02 0.04 0.08 0.16 0.32];

%% -------- Compute Response Tensor Resp(ori,sf,cell) ----------
nOri = numel(oris); nSF = numel(sfs);
Resp = nan(nOri,nSF,nCell);

for io=1:nOri
    for sf=1:nSF
        rows = (stim.orientation==oris(io)) & (stim.spatial_frequency==sfs(sf));
        starts = stim.("start")(rows); ends_ = stim.("end")(rows);

        if any(starts==0)
            starts = starts+1; ends_=ends_+1;
        end

        starts = max(starts,1);
        ends_  = min(ends_,T_f);
        nTrials = numel(starts);
        trial_mean = nan(nCell,nTrials);

        for tr=1:nTrials
            sIdx = starts(tr); eIdx = ends_(tr);
            bStart = max(1, sIdx - round(1/dt));
            bEnd   = sIdx-1;

            baseline = mean(F(:,bStart:bEnd),2,'omitnan');
            stim_mean = mean(F(:,sIdx:eIdx),2,'omitnan');
            trial_mean(:,tr) = stim_mean - baseline;
        end

        Resp(io,sf,:) = mean(trial_mean,2,'omitnan');
    end
end

fprintf('✅ Resp tensor computed: [%d ori × %d sf × %d cells]\n',nOri,nSF,nCell);

%% -------- Reduce to Allen bins (6×5) --------
Resp_fixed = nan(length(ori_expected), length(sf_expected), nCell);

for io=1:nOri
    oi = find(ori_expected==oris(io));
    for sf=1:nSF
        si = find(sf_expected==sfs(sf));
        if ~isempty(oi) && ~isempty(si)
            Resp_fixed(oi,si,:) = Resp(io,sf,:);
        end
    end
end

fprintf('✅ Resp_fixed size = [%d × %d × %d]\n', size(Resp_fixed));

%% -------- Select ONE cell for detailed view --------
cell_idx = 1; % <---- change to explore others

%% -------- Extract trial-by-trial traces for one cell --------
ori_choice   = 60;
sf_choice    = 0.08;
phase_choice = 0.5;

rows = (stim.orientation==ori_choice) & (stim.spatial_frequency==sf_choice) & (stim.phase==phase_choice);
starts = stim.("start")(rows); ends_ = stim.("end")(rows);

if any(starts==0)
    starts=starts+1; ends_=ends_+1;
end
starts=max(starts,1); ends_=min(ends_,T_f);

trace = F(cell_idx,:);
nTrials = numel(starts);
fprintf('Trials for cell %d: %d\n', cell_idx, nTrials);

trial_abs = cell(nTrials,1);
trial_tr  = cell(nTrials,1);
trial_len = zeros(nTrials,1);

for tr=1:nTrials
    sIdx = starts(tr); eIdx=ends_(tr);
    mask=false(1,T_f); mask(sIdx:eIdx)=true;
    trial_abs{tr}=time_s(mask);
    trial_tr{tr}=trace(mask);
    trial_len(tr)=eIdx-sIdx+1;
end

%% -------- Build aligned matrix & mean trace --------
max_len = max(trial_len);
t_align = (0:max_len-1)*dt;
trial_mat = nan(nTrials,max_len);
for tr=1:nTrials
    y=trial_tr{tr}; trial_mat(tr,1:numel(y))=y;
end
mean_resp = mean(trial_mat,1,'omitnan');

%% -------- PLOT: Single-cell trial stack + mean + heatmap --------
fig1 = figure('Color','w','Position',[50 50 1400 800]);

max_show = min(10,nTrials);
left_x=0.05; left_w=0.45; vgap=0.01;
top_y=0.95; bottom_y=0.05;
row_h=(top_y-bottom_y-(max_show-1)*vgap)/max_show;

allY = cell2mat(cellfun(@(x)x(:),trial_tr(1:max_show),'UniformOutput',false));
yl = [min(allY) max(allY)] + [-1 1]*0.05*range(allY);

for tr=1:max_show
    ax=subplot('Position',[left_x, top_y-tr*row_h-(tr-1)*vgap, left_w, row_h]);
    plot(trial_abs{tr}, trial_tr{tr},'k','LineWidth',1); ylim(yl); grid on
    if tr<max_show, set(ax,'XTickLabel',[]); else, xlabel('Absolute time (s)'); end
    ylabel(sprintf('Trial %d',tr));
end

ax_avg = subplot('Position',[0.55,0.56,0.4,0.38]);
plot(t_align,mean_resp,'r','LineWidth',2); grid on
xlabel('Time from onset (s)'); ylabel('Mean dF/F');
title(sprintf('Cell %d (ID %d) Mean | Exp %d', ...
    cell_idx,cell_ids(cell_idx),exp_id));

ax_hm = subplot('Position',[0.55,0.08,0.4,0.38]);
imagesc(t_align,1:nTrials,trial_mat);
colormap(ax_hm,parula); colorbar
xlabel('Time from onset (s)'); ylabel('Trial #');
title('Trial × Time Heatmap');

fprintf('✅ Single-cell figure complete\n');
disp('Press any key to continue...');
pause;

%% -------- Single cell tuning heatmap (6×5 matrix) --------
resp_mat = squeeze(Resp_fixed(:,:,cell_idx));

figure('Color','w','Position',[900 100 450 400]);
imagesc(resp_mat);
colormap(parula); colorbar;
axis square tight;

set(gca,'XTick',1:length(sf_expected), 'XTickLabel',string(sf_expected));
set(gca,'YTick',1:length(ori_expected),'YTickLabel',string(ori_expected));

xlabel('Spatial Frequency (cycles/°)');
ylabel('Orientation (°)');

title(sprintf('Exp %d | Cell %d (ID %d)\nStatic Gratings Tuning (6×5)', ...
      exp_id, cell_idx, cell_ids(cell_idx)), ...
      'FontWeight','bold','FontSize',12);

disp('Press any key to continue...');
pause;

%% -------- 3D Clustering & Tuning Cube --------
X = zeros(nCell, numel(ori_expected)*numel(sf_expected));
for c = 1:nCell
    tmp = Resp_fixed(:,:,c);
    X(c,:) = tmp(:).';
end

k = 6;
[idxK,~] = kmeans(X,k,'Distance','correlation','Replicates',30);

[~,prefOri] = max(squeeze(max(Resp_fixed,[],2)),[],1);
[~,prefSF]  = max(squeeze(max(Resp_fixed,[],1)),[],1);

[~,order] = sortrows([idxK(:),prefOri(:),prefSF(:)]);
Resp_sorted = Resp_fixed(:,:,order);

[yy,xx,zz] = ndgrid(1:length(ori_expected),1:length(sf_expected),1:nCell);
vals = Resp_sorted(:);

fig2=figure('Color','w','Position',[200 100 1100 750]);
scatter3(xx(:),yy(:),zz(:),50,vals,'filled');
colormap(parula); colorbar; grid on; box on;
view(45,30); alpha(0.8);
xlabel('SF index'); ylabel('Ori index'); zlabel('Cell (sorted)');
title(sprintf('Exp %d | 3D Tuning Cube | %d Cells | k=%d',exp_id,nCell,k));
set(gca,'XTick',1:length(sf_expected),'XTickLabel',string(sf_expected));
set(gca,'YTick',1:length(ori_expected),'YTickLabel',string(ori_expected));
caxis([min(vals) max(vals)*0.9]);

fprintf('✅ 3D tuning cube complete\n');
disp('Press any key to continue...');
pause;

%% =============================================
% Population Tuning Heatmaps
% =============================================

assert(exist('Resp_fixed','var')==1,'Resp_fixed missing');
assert(exist('ori_expected','var')==1,'ori_expected missing');
assert(exist('sf_expected','var')==1,'sf_expected missing');

[nOri, nSF, nCell] = size(Resp_fixed);
fprintf('Resp_fixed = [%d ori × %d sf × %d cells]\n', nOri, nSF, nCell);

%% --------------------------------------------------------------------
% 1) Concatenated heatmap: cells × (ORI × SF)
%% --------------------------------------------------------------------
wide_mat = reshape(Resp_fixed, [nOri*nSF, nCell]).';

figure('Color','w','Position',[100 100 1200 400]);
imagesc(wide_mat);
colormap(parula); colorbar;
title('Population Tuning: Cells × (Orientation × Spatial Frequency)');
xlabel('Orientation blocks (each ORI contains all SFs)');
ylabel('Cells');

xt = 1:nSF:(nOri*nSF);
set(gca,'XTick',xt,'XTickLabel',string(ori_expected));
xlabel('Orientation (°) — each block has SFs: ' + join(string(sf_expected)));

disp('Press any key to continue...');
pause;

%% --------------------------------------------------------------------
% 2) Concatenated heatmap: cells × (SF × ORI)
%% --------------------------------------------------------------------
wide_mat_sf = permute(Resp_fixed, [2 1 3]);
wide_mat_sf = reshape(wide_mat_sf, [nSF*nOri, nCell]).';

figure('Color','w','Position',[100 550 1200 400]);
imagesc(wide_mat_sf);
colormap(parula); colorbar;
title('Population Tuning: Cells × (Spatial Frequency × Orientation)');
xlabel('SF blocks (each block contains all ORIs)');
ylabel('Cells');

xt = 1:nOri:(nSF*nOri);
set(gca,'XTick',xt,'XTickLabel',string(sf_expected));
xlabel('Spatial Frequency (cycles/°) — each block has ORIs: ' + join(string(ori_expected)));

disp('Press any key to continue...');
pause;

%% ======================================
% PCA ANALYSIS FOR ORI & SF LOOPS
%% ======================================

%% ---- Prepare variables for PCA ----
% (this part is unchanged from your code)
% ...
%% ======================================
% REQUIREMENTS in workspace BEFORE running:
% Resp, ori_expected, sf_expected, stim, F, time_s, dt
%% ======================================

%% ---- Validate Workspace ----
req = {'Resp','ori_expected','sf_expected','stim','F','time_s','dt'};
for r = req
    if evalin('base',sprintf('exist(''%s'',''var'')',r{1}))~=1
        error('Missing %s in workspace', r{1});
    end
end

Resp           = evalin('base','Resp');
ori_expected   = evalin('base','ori_expected');
sf_expected    = evalin('base','sf_expected');
stim           = evalin('base','stim');
F              = evalin('base','F');
time_s         = evalin('base','time_s');
dt             = evalin('base','dt');

%% ---- Dimensions ----
nOri = length(ori_expected);
nSF  = length(sf_expected);
nCell = size(F,1);

% Extract first 4 phase values
all_phs   = unique(stim.phase);
phase_vals = all_phs(1:4);
nPhase = 4;

assert(nOri==6,"Need 6 orientations");
assert(nSF==5,"Need 5 spatial freqs");
assert(nPhase==4,"Need 4 phases");

%% (Your PCA loop + figures come next)
%% ---- Allocate ----
Resp_phase = nan(nOri, nSF, nPhase, nCell);

fprintf('\n⏳ Computing phase-resolved ΔF/F ...\n');

%% ---- Compute Response per ORI × SF × PHASE × CELL ----
for io = 1:nOri
for isf = 1:nSF
for iph = 1:nPhase

    rows = stim.orientation==ori_expected(io) & ...
           stim.spatial_frequency==sf_expected(isf) & ...
           stim.phase==phase_vals(iph);

    if ~any(rows), continue; end

    starts = stim.start(rows);
    ends_  = stim.end(rows);

    for c = 1:nCell
        vals = [];
        for tr = 1:numel(starts)
            s = max(1,starts(tr));
            e = min(length(time_s),ends_(tr));
            base_idx = max(1,s-round(1/dt)):s-1;
            baseline = mean(F(c,base_idx),'omitnan');
            vals(end+1) = mean(F(c,s:e)) - baseline;
        end
        Resp_phase(io,isf,iph,c) = mean(vals,'omitnan');
    end
end
end
end

fprintf('✅ Phase matrix: [%d ori × %d sf × %d phase × %d cells]\n', ...
        nOri, nSF, nPhase, nCell);
%% ---- Avg Across Phase ----
Resp_avg = squeeze(mean(Resp_phase,3)); % 6×5×cells

%% ---- Flatten to stimulus × cell ----
stimMat = reshape(Resp_avg,[nOri*nSF, nCell]); % 30 × cells
stimMat_z = zscore(stimMat, 0, 2);

%% ---- PCA (2 components) ----
[coeff,score,~,~,expl] = pca(stimMat_z');

stimPC = score(:,1:2); % Only 2 PCs

fprintf('⚡ PC1 %.1f%% | PC2 %.1f%% variance explained\n', expl(1), expl(2));

%% ---- Index Labels ----
ori_idx = repelem((1:nOri)', nSF); % 30×1
sf_idx  = repmat((1:nSF)', nOri, 1); % 30×1

%% ---- Axis Limits ----
xmin = min(stimPC(:,1)); xmax = max(stimPC(:,1));
ymin = min(stimPC(:,2)); ymax = max(stimPC(:,2));

%% ---- Viz Parameters ----
alphaLine   = 0.35;
alphaMarker = 0.6;
lw = 2;
ms = 8;

sf_colors  = lines(nSF);
ori_colors = hsv(nOri);

disp('Press any key to continue to PCA plots...');
pause;


%% ===============================================================
% IMPROVED PCA VISUALIZATION — With explained variance & cell activity
% ===============================================================

% Compute percentage for axes
pc1_var = expl(1);
pc2_var = expl(2);
fprintf('⚡ PC1 = %.2f%% | PC2 = %.2f%% of total variance\n', pc1_var, pc2_var);

% Compute per-stimulus "activity" magnitude = mean(|ΔF/F|) across cells
stim_activity = mean(abs(stimMat_z), 2, 'omitnan');
stim_activity = rescale(stim_activity, 10, 100);  % scale for marker size

%% -------------------------------
% FIGURE 1 — Orientation loops (per SF)
%% -------------------------------
figure('Color','w','Position',[50 80 1500 800]);
tiledlayout(2,ceil((nSF+1)/2),'TileSpacing','compact','Padding','compact')

for sf = 1:nSF
    nexttile; hold on;
    idx = (sf_idx == sf);
    pts = stimPC(idx,:);
    act = stim_activity(idx);

    % Plot path
    plot(pts(:,1), pts(:,2), '-', ...
        'Color',[sf_colors(sf,:) 0.35], 'LineWidth',2);

    % Scatter points with activity weighting
    scatter(pts(:,1), pts(:,2), act, ...
        'MarkerFaceColor',sf_colors(sf,:), ...
        'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.7);

    title(sprintf('SF = %.2f cpd', sf_expected(sf)));
    xlabel(sprintf('PC1 (%.1f%%)', pc1_var));
    ylabel(sprintf('PC2 (%.1f%%)', pc2_var));
    grid on; axis([xmin xmax ymin ymax]); axis equal;
end

% Overlay subplot
nexttile; hold on;
for sf = 1:nSF
    idx = (sf_idx == sf);
    pts = stimPC(idx,:);
    act = stim_activity(idx);
    plot(pts(:,1), pts(:,2), '-', 'Color',[sf_colors(sf,:) 0.3], 'LineWidth',2);
    scatter(pts(:,1), pts(:,2), act, ...
        'MarkerFaceColor',sf_colors(sf,:), ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.6);
end
title("Overlay: ORI trajectories across SF");
xlabel(sprintf('PC1 (%.1f%%)', pc1_var));
ylabel(sprintf('PC2 (%.1f%%)', pc2_var));
grid on; axis([xmin xmax ymin ymax]); axis equal;
legend(string(sf_expected),'Location','bestoutside');
sgtitle("2-D PCA — Orientation Sweeps Across Spatial Frequencies (Activity-weighted)", ...
    "FontWeight","bold");

disp('Press any key to continue to PCA plots...');
pause;

%% -------------------------------
% FIGURE 2 — SF loops (per ORI)
%% -------------------------------
figure('Color','w','Position',[50 920 1500 800]);
tiledlayout(2,ceil((nOri+1)/2),'TileSpacing','compact','Padding','compact')

for ori = 1:nOri
    nexttile; hold on;
    idx = (ori_idx == ori);
    pts = stimPC(idx,:);
    act = stim_activity(idx);

    plot(pts(:,1), pts(:,2), '-', 'Color',[ori_colors(ori,:) 0.4], 'LineWidth',2);
    scatter(pts(:,1), pts(:,2), act, ...
        'MarkerFaceColor',ori_colors(ori,:), ...
        'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.7);

    title(sprintf('ORI = %d°', ori_expected(ori)));
    xlabel(sprintf('PC1 (%.1f%%)', pc1_var));
    ylabel(sprintf('PC2 (%.1f%%)', pc2_var));
    grid on; axis([xmin xmax ymin ymax]); axis equal;
end

nexttile; hold on;
for ori = 1:nOri
    idx = (ori_idx == ori);
    pts = stimPC(idx,:);
    act = stim_activity(idx);
    plot(pts(:,1), pts(:,2), '-', 'Color',[ori_colors(ori,:) 0.35], 'LineWidth',2);
    scatter(pts(:,1), pts(:,2), act, ...
        'MarkerFaceColor',ori_colors(ori,:), ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.6);
end
title("Overlay: SF trajectories across ORI");
xlabel(sprintf('PC1 (%.1f%%)', pc1_var));
ylabel(sprintf('PC2 (%.1f%%)', pc2_var));
grid on; axis([xmin xmax ymin ymax]); axis equal;
legend(string(ori_expected),'Location','bestoutside');
sgtitle("2-D PCA — Spatial Frequency Sweeps Across Orientations (Activity-weighted)", ...
    "FontWeight","bold");

fprintf('✅ PCA visualization with variance + activity weighting complete.\n');

 
