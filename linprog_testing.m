% Parameters
epsilons = 0.01:0.05:0.5;  % Cannot start from 0 to avoid division by zero
dimensions = 10:10:50;
sizes = 10000:10000:50000;
n_repeats = 100;

% LP options
linprog_options = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'interior-point');

% Initialize result table
results_table = [];

for dim = dimensions
    for sz = sizes
        for eps = epsilons
            correct_count = 0;

            for trial = 1:n_repeats
                % Generate data with epsilon label noise
                [X_all, y_all] = Generate_CLS(sz, dim, eps);  % X: n x d, y: n x 1
                X_all = X_all';
                y_all = y_all';

                n = size(X_all, 2);

                % Determine sample size based on epsilon
                sample_size = ceil(dim / eps);
                sample_size = min(sample_size, n);

                indices = randperm(n, sample_size);
                X = X_all(:, indices);
                y = y_all(:, indices);

                % Construct A matrix for LP
                A = -y' .* [X', ones(sample_size, 1)];
                b_vec = -ones(sample_size, 1);
                f = zeros(dim + 1, 1);  % Arbitrary objective

                % Solve LP
                [~, ~, exitflag, ~] = linprog(f, A, b_vec, [], [], [], [], linprog_options);

                % Determine expected result: epsilon = 0 → LS, else → not exactly LS
                expected_separable = (eps == 0);
                sample_separable = (exitflag == 1);

                if sample_separable == expected_separable
                    correct_count = correct_count + 1;
                end
            end

            % Save result
            results_table = [results_table; dim, sz, eps, sample_size, correct_count];

            fprintf('dim = %d, size = %d, epsilon = %.2f, sample_size = %d, correct = %d/%d\n', ...
                dim, sz, eps, sample_size, correct_count, n_repeats);
        end
    end
end

% Convert to table and save
T = array2table(results_table, ...
    'VariableNames', {'dimension', 'size', 'epsilon', 'sample_size', 'correct_guesses'});

writetable(T, 'CLS_LP_accuracy_with_sampling.csv');
fprintf('✅ Results saved to CLS_LP_accuracy_with_sampling.csv\n');
