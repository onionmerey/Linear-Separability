% Settings
data_folder = 'C:\\Users\User\Desktop\Research\Datasets_3D_new';  % <-- set your path
epsilons = 0.01:0.05:0.5;
n_repeats = 100;
d = 3;  % Dimensionality

% LP options
lp_options = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'interior-point');

% Result accumulator
results = [];

% List all CSV files
files = dir(fullfile(data_folder, '*.csv'));

for file = files'
    filename = file.name;
    filepath = fullfile(data_folder, filename);
    disp(['Processing ', filename]);

    % Parse the separability degree from filename
    parts = split(filename, '_');
    if numel(parts) < 3
        warning('Skipping malformed filename: %s', filename);
        continue;
    end

    sep_degree = str2double(parts{2});
    true_separable = (sep_degree == 0.0);  % Only exactly 0 is considered linearly separable

    % Read the CSV
    data = readmatrix(filepath);  % (n x 4)
    X_all = data(:, 1:3);         % features
    y_all = data(:, 4);           % labels in {0, 1}

    y_all = 2 * y_all - 1;  % convert to {-1, +1}
    n = size(X_all, 1);

    for eps = epsilons
        sample_size = min(2*ceil(d / eps), n);
        correct = 0;

        for i = 1:n_repeats
            idx = randperm(n, sample_size);
            X = X_all(idx, :);
            y = y_all(idx);

            % Build LP constraints: -yᵢ(wᵗx + b) ≤ -1
            A = -y .* [X, ones(sample_size, 1)];
            b_vec = -ones(sample_size, 1);
            f = zeros(d + 1, 1);

            [~, ~, exitflag] = linprog(f, A, b_vec, [], [], [], [], lp_options);
            is_separable = (exitflag == 1);

            if is_separable == true_separable
                correct = correct + 1;
            end
        end

        % Log result
        results = [results; {filename, sep_degree, eps, sample_size, correct}];
        fprintf('  eps = %.3f | sample = %d | correct = %d/%d\n', eps, sample_size, correct, n_repeats);
    end

    fprintf('----------------------------------\n');
end

% Save results
T = cell2table(results, 'VariableNames', ...
    {'filename', 'sep_degree', 'epsilon', 'sample_size', 'correct_guesses'});
writetable(T, 'lp_3D_coefficient2_results.csv');
fprintf('✅ Results saved to lp_3D_results.csv\n');
