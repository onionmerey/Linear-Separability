function COVERTYPE_LP_AllPairs()
% COVERTYPE_LP_AllPairs
% - Loads UCI Covertype dataset (54 features, 581k samples)
% - For each class pair a<b:
%       * Ground truth separability via full LP feasibility
%       * Uniform subsampling LP tests with rule m = max(p+1, min(n, ceil(p/eps)))
% - Saves incremental results into CSV files
%
% Requires: Optimization Toolbox (linprog)

clc; clear; close all; tic;

%% Config
epsilons   = [0.50 0.45 0.4 0.35 0.3 0.25 0.20 0.15 0.10 0.05];
R          = 50;                 % trials per epsilon (reduce for speed)
rng(42);                        % reproducibility

data_file   = 'covtype.data';   % UCI file (download & unzip beforehand)
results_dir = fullfile(pwd,'results'); 
if ~exist(results_dir,'dir'), mkdir(results_dir); end

summary_csv = fullfile(results_dir, 'covtype_full_lp_summary.csv');
sample_csv  = fullfile(results_dir, 'covtype_lp_sampling_results.csv');

lp_opts = optimoptions('linprog','Display','none','Algorithm','interior-point');

%% 1) Load Covertype dataset
fprintf('Loading %s ...\n', data_file);
raw = readmatrix(data_file, 'FileType','text');
Xall = raw(:,1:end-1);   % n x 54
yall = raw(:,end);       % n x 1 labels (1..7)

classes = unique(yall);
pairs   = nchoosek(classes,2);
fprintf('Loaded Covertype: n=%d, p=%d, classes=%s\n', size(Xall,1), size(Xall,2), mat2str(classes));

%% 2) Loop over all class pairs
for k = 1:size(pairs,1)
    a = pairs(k,1); b = pairs(k,2);
    fprintf('\n=== Pair %d/%d: %d vs %d ===\n', k, size(pairs,1), a, b);

    % Extract samples
    idxA = find(yall == a);
    idxB = find(yall == b);
    X = [Xall(idxA,:); Xall(idxB,:)];   % n x p
    y = [ones(numel(idxA),1)*-1; ones(numel(idxB),1)]; % -1 for a, +1 for b

    n = size(X,1); p = size(X,2);
    fprintf('n=%d, p=%d | counts: %d=%d, %d=%d\n', n, p, a, numel(idxA), b, numel(idxB));

    %% Ground truth full LP feasibility
    t0 = tic;
    [is_sep_full, exit_full] = lp_separability_feasibility(X, y, lp_opts);
    full_lp_time = toc(t0);

    fprintf('Full LP: separable=%d (exitflag=%d), time=%.2fs\n', ...
        is_sep_full, exit_full, full_lp_time);

    % Save summary row
    Summary = table( ...
        string(sprintf('%d_vs_%d', a, b)), a, b, n, p, numel(idxA), numel(idxB), ...
        is_sep_full, exit_full, full_lp_time, ...
        'VariableNames', {'pair','class_neg','class_pos','n','p','n_neg','n_pos', ...
                          'separable','exitflag','full_lp_time_sec'});
    append_table_csv(Summary, summary_csv);

    %% Uniform sampling LP tests
    for ei = 1:numel(epsilons)
        eps = epsilons(ei);
        m   = max(p+1, min(n, 10*ceil(p/eps)));

        correct    = false(R,1);
        samp_times = zeros(R,1);

        for r = 1:R
            idx = randperm(n,m);
            t1 = tic;
            [is_sep_sample, ~] = lp_separability_feasibility(X(idx,:), y(idx), lp_opts);
            samp_times(r) = toc(t1);
            correct(r)    = (is_sep_sample == is_sep_full);
        end

        acc   = mean(correct);
        avg_t = mean(samp_times);

        Row = table( ...
            string(sprintf('%d_vs_%d', a, b)), a, b, n, p, eps, m, R, ...
            is_sep_full, full_lp_time, acc, avg_t, ...
            'VariableNames', {'pair','class_neg','class_pos','n','p', ...
                              'epsilon','m','repeats', ...
                              'ground_truth_separable','full_lp_time_sec', ...
                              'accuracy','avg_sample_time_sec'});
        append_table_csv(Row, sample_csv);

        fprintf('  eps=%.2f m=%d | acc=%.2f | avg_t=%.3fs\n', eps, m, acc, avg_t);
    end
end

fprintf('\nDone.\nSummaries:   %s\nPer-epsilon: %s\n', summary_csv, sample_csv);
toc;

end

%% ======================= Helpers =======================
function [is_sep, exitflag] = lp_separability_feasibility(X, y, linprog_options)
    % Check feasibility of linear separator via LP
    [n,p] = size(X);
    A = - (y .* [X, ones(n,1)]);   % elementwise multiply instead of diag
    b = -ones(n,1);
    f = zeros(p+1,1);              % dummy objective
    [~,~,exitflag] = linprog(f, A, b, [], [], [], [], linprog_options);
    is_sep = (exitflag == 1);
end


function append_table_csv(T, filename)
% Append a row to CSV file (create if missing)
if exist(filename,'file')
    try
        writetable(T, filename, 'WriteMode','append');
    catch
        U = readtable(filename);
        U = [U; T]; %#ok<AGROW>
        writetable(U, filename);
    end
else
    writetable(T, filename);
end
end
