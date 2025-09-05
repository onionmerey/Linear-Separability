function EvaluateLP_OnGenerated2D_MultiN()
% CSV columns (exact):
%   num_points, true_distance, epsilon, sample_size, Method1_avg_time, Method2_avg_time, Method1_accuracy, Method2_accuracy
%
% This version generates a separable 2D dataset for each N in Ns and
% performs the same sampling + testing routine (Method 1 and Method 2).
%
% Run:
%   EvaluateLP_OnGenerated2D_MultiN;

clc; close all;

%% ==== USER PARAMS ====
Ns          = 5000:5000:100000;         % Ns from 5k to 100k
epsilons    = 0.50:-0.05:0.01;          % same as before
n_repeats   = 100;                       % you can lower for speed
k_factor    = 5;                         % k in k*d/epsilon
lp_opts     = optimoptions('linprog','Display','none','Algorithm','interior-point');
rng('shuffle');

rows = {};  % will collect all (N, epsilon) rows here

%% ==== MAIN LOOP OVER Ns ====
for N = Ns
    %% ---- Generate 2D Data (only input part changed) ----
    mergedData = Generate_LS_data(2, N);   % points are columns; row1=labels {0,1}
    labels01   = mergedData(1, :).';
    X          = mergedData(2:3, :).';     % N x 2
    y          = ones(N,1);  y(labels01==0) = -1;  % map 0->-1, 1->+1

    p     = 2;
    dvars = p + 1;                          % [w1 w2 b]
    Z     = [X, ones(N,1)];
    tol   = 1e-8;

    % ---- Ground truth via full-data LP feasibility ----
    % y_i*(w^T x_i + b) >= 1   <=>   -y .* (Z*theta) <= -1
    A_full = -Z .* y;                       % Nx3; implicit expansion
    b_full = -ones(N,1);
    f0     = zeros(dvars,1);

    [theta_full,~,exitflag_full] = linprog(f0, A_full, b_full, [], [], [], [], lp_opts);
    gt_sep = (exitflag_full == 1);

    % A numeric "true_distance": use lower-bound margin 1/||w||2 when separable, else 0
    if gt_sep && ~isempty(theta_full)
        w = theta_full(1:2);
        true_distance = 1 / norm(w);
    else
        true_distance = 0.0;                % (shouldn't happen with this generator)
    end

    fprintf('==== N = %d | true_distance ~ %.6g ====\n', N, true_distance);

    %% ---- Experiments across epsilons (same routine) ----
    for eps = epsilons
        % Common initial sample size for both methods:
        r  = min(N, max(p, ceil(k_factor * p / eps)));
        Tv = ceil(2 / eps);                 % Method 2 verification size

        %% ---------------- Method 1: One-shot LP ----------------
        correct1 = 0; total_t1 = 0;
        for rep = 1:n_repeats
            t0   = tic;
            Ridx = randperm(N, r);
            A_R  = A_full(Ridx, :);
            b_R  = b_full(Ridx);

            [~,~,exitR] = linprog(f0, A_R, b_R, [], [], [], [], lp_opts);
            accept1     = (exitR == 1);

            total_t1 = total_t1 + toc(t0);
            if accept1 == gt_sep, correct1 = correct1 + 1; end
        end
        m1_avg_t = total_t1 / n_repeats;
        m1_acc   = 100 * correct1 / n_repeats;

        %% ------------- Method 2: LP + verify 2/eps -------------
        correct2 = 0; total_t2 = 0;
        for rep = 1:n_repeats
            t0   = tic;

            % Initial LP on r points
            Ridx = randperm(N, r);
            A_R  = A_full(Ridx, :);
            b_R  = b_full(Ridx);
            [thetaR,~,exitR] = linprog(f0, A_R, b_R, [], [], [], [], lp_opts);

            if exitR ~= 1 || isempty(thetaR)
                accept2 = false;
            else
                % Verify on fresh points (disjoint if possible)
                pool = setdiff(1:N, Ridx, 'stable');
                if numel(pool) >= Tv
                    Yidx = pool(randperm(numel(pool), Tv));
                else
                    Yidx = randi(N, Tv, 1); % fallback
                end
                lhs     = (Z(Yidx,:)*thetaR) .* y(Yidx);
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

        fprintf('N=%d | eps=%.3f | r=%d | M1: %.3fs %.1f%% | M2: %.3fs %.1f%%\n', ...
            N, eps, r, m1_avg_t, m1_acc, m2_avg_t, m2_acc);
    end

    fprintf('------------------------------------------------------\n');
end

%% ==== SAVE RESULTS (all Ns) ====
T = cell2table(rows, 'VariableNames', ...
    {'num_points','true_distance','epsilon','sample_size', ...
     'Method1_avg_time','Method2_avg_time','Method1_accuracy','Method2_accuracy'});

out_csv = fullfile(pwd, sprintf('LP_M1_vs_M2_generated_multiN_k%d_%s.csv', ...
    k_factor, datestr(now,'yyyymmdd_HHMMSS')));
writetable(T, out_csv);
fprintf('✅ Results saved to %s\n', out_csv);

end % main

% ========================================================================
% Your provided generator (unchanged). Keeps two linearly separable sets.
% ========================================================================
function [mergedData] = Generate_LS_data(n, m)
    % 1. Randomly generate m points in n-dimensional space with values ranging from -1000 to 1000
    points = randi([-1000, 1000], n, m);

    % 2. Generate a random normal vector
    normal_vector = randn(n, 1);
    normal_vector = normal_vector / norm(normal_vector); % Normalize

    % 3. Calculate the distance of each point to the hyperplane P
    distances = sum(points .* normal_vector, 1)';

    % 4. Divide into two sets using the hyperplane (closest m/4 in A)
    [~, sorted_indices] = sort(distances);
    A_indices = sorted_indices(1:floor(m/4));
    B_indices = sorted_indices(floor(m/4)+1:end);

    A = points(:, A_indices);
    B = points(:, B_indices);

    % 5. Shift points in A and B away from the hyperplane
    shift_value = 5;
    A = A - normal_vector * shift_value;
    B = B + normal_vector * shift_value;

    % 6. Merge and label (0 for A, 1 for B); points are columns
    labels_A = zeros(1, size(A, 2));
    labels_B = ones(1, size(B, 2));
    mergedData = [labels_A, labels_B; A, B];
end
