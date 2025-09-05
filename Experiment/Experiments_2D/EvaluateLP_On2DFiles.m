function EvaluateLP_On2DFiles()
% CSV columns (exact):
%   num_points, true_distance, epsilon, sample_size, Method1_avg_time, Method2_avg_time, Method1_accuracy, Method2_accuracy
%
% Filename example: 2D_60000_0.280233333333_42137956
%   -> true_distance parsed as the 3rd numeric token (since "2D" contributes the first numeric '2').

clc; close all;

%% ==== USER PARAMS ====
data_dir    = fullfile(pwd, 'Datasets_2D');  % <- change if needed
file_glob   = '*';                           % e.g. '2D_*'
epsilons    = 0.50:-0.05:0.01;
n_repeats   = 100;

k_factor    = 2;  % k in k*d/epsilon
lp_opts = optimoptions('linprog', 'Display','none', 'Algorithm','interior-point');
rng('shuffle');

%% ==== DISCOVER FILES ====
files = dir(fullfile(data_dir, file_glob));
if isempty(files)
    error('No files found in "%s" matching "%s".', data_dir, file_glob);
end

rows = {};

for fidx = 1:numel(files)
    fname = files(fidx).name;
    fpath = fullfile(files(fidx).folder, fname);

    try
        [X, y] = read_2d_labeled_file(fpath);   % X: N×2, y: N×1 in {+1,-1}
    catch ME
        warning('Skipping "%s": %s', fname, ME.message);
        continue;
    end

    N = size(X,1);
    p = 2;
    dvars = p+1;                   % variables: [w1 w2 b]
    Z = [X, ones(N,1)];
    tol = 1e-8;

    % ---- true distance from filename (3rd numeric token) ----
    true_distance = parse_distance_from_fname_tokens(fname);

    % ---- Ground truth via full-data LP feasibility ----
    % y_i*(w^T x_i + b) >= 1   <=>   -y .* (Z*theta) <= -1
    A_full = -Z .* y;            % Nx3 (implicit expansion)
    b_full = -ones(N,1);
    f0     = zeros(dvars,1);
    [~,~,exitflag_full] = linprog(f0, A_full, b_full, [], [], [], [], lp_opts);
    gt_sep = (exitflag_full == 1);

    for eps = epsilons
        % Common initial sample size for both methods:
        r = min(N, max(p, ceil(k_factor * p / eps)));
        Tv = ceil(2 / eps);  % Method 2 verification size

        %% ---------------- Method 1: One-shot LP ----------------
        correct1 = 0; total_t1 = 0;
        for rep = 1:n_repeats
            t0 = tic;
            Ridx = randperm(N, r);
            A_R  = A_full(Ridx, :);
            b_R  = b_full(Ridx);

            [~,~,exitR] = linprog(f0, A_R, b_R, [], [], [], [], lp_opts);
            accept1 = (exitR == 1);

            total_t1 = total_t1 + toc(t0);
            if accept1 == gt_sep, correct1 = correct1 + 1; end
        end
        m1_avg_t = total_t1 / n_repeats;
        m1_acc   = 100 * correct1 / n_repeats;

        %% ------------- Method 2: LP + verify 2/eps -------------
        correct2 = 0; total_t2 = 0;
        for rep = 1:n_repeats
            t0 = tic;

            % Initial LP on r points
            Ridx = randperm(N, r);
            A_R  = A_full(Ridx, :);
            b_R  = b_full(Ridx);
            [theta,~,exitR] = linprog(f0, A_R, b_R, [], [], [], [], lp_opts);

            if exitR ~= 1 || isempty(theta)
                accept2 = false;
            else
                % Verify on fresh points (disjoint if possible)
                pool = setdiff(1:N, Ridx, 'stable');
                if numel(pool) >= Tv
                    perm = randperm(numel(pool), Tv);
                    Yidx = pool(perm(:));
                else
                    Yidx = randi(N, Tv, 1);  % fallback
                end
                lhs = (Z(Yidx,:)*theta) .* y(Yidx);
                accept2 = all(lhs >= 1 - tol);
            end

            total_t2 = total_t2 + toc(t0);
            if accept2 == gt_sep, correct2 = correct2 + 1; end
        end
        m2_avg_t = total_t2 / n_repeats;
        m2_acc   = 100 * correct2 / n_repeats;

        % ---- Store single row with both methods ----
        rows(end+1,:) = { ...
            N, true_distance, eps, r, ...
            m1_avg_t, m2_avg_t, m1_acc, m2_acc}; %#ok<AGROW>

        fprintf('File=%s | N=%d | d=%.6g | eps=%.3f | r=%d | M1: %.3fs %.1f%% | M2: %.3fs %.1f%%\n', ...
            fname, N, true_distance, eps, r, m1_avg_t, m1_acc, m2_avg_t, m2_acc);
    end

    fprintf('------------------------------------------------------\n');
end

%% ==== SAVE RESULTS ====
T = cell2table(rows, 'VariableNames', ...
    {'num_points','true_distance','epsilon','sample_size', ...
     'Method1_avg_time','Method2_avg_time','Method1_accuracy','Method2_accuracy'});

out_csv = fullfile(pwd, sprintf('LP_M1_vs_M2_k%d_%s.csv', k_factor, datestr(now,'yyyymmdd_HHMMSS')));
writetable(T, out_csv);
fprintf('✅ Results saved to %s\n', out_csv);

end % main


%% ---------- Helpers ----------

function [X, y] = read_2d_labeled_file(fpath)
fid = fopen(fpath,'r'); if fid < 0, error('Cannot open: %s', fpath); end
cleanup = onCleanup(@() fclose(fid));
N = sscanf(strtrim(fgetl(fid)), '%d', 1);
if isempty(N) || N<=0, error('Bad first line (N) in %s', fpath); end
X = zeros(N,2); y = zeros(N,1);
for i = 1:N
    line = fgetl(fid); if ~ischar(line), error('EOF at row %d in %s', i, fpath); end
    tok = textscan(line, '%f %f %s');
    if any(cellfun(@isempty, tok)), error('Bad row %d in %s: "%s"', i, fpath, line); end
    X(i,:) = [tok{1}, tok{2}];
    switch char(tok{3}{1})
        case '.', y(i) = -1;
        case '#', y(i) = +1;
        otherwise, error('Unknown label at row %d in %s', i, fpath);
    end
end
end

function dval = parse_distance_from_fname_tokens(fname)
% "2D_60000_0.280233333333_42137956" -> numerics: [2, 60000, 0.280233..., 42137956]
% We want 0.280233..., which is the 3rd numeric token.
nums = regexp(fname, '([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', 'tokens');
if isempty(nums) || numel(nums) < 3
    dval = NaN;
else
    dval = str2double(nums{3}{1});
end
end
