function [X, y] = Generate_CLS(n_points, dimension, epsilon)
% Generate_CLS creates a nearly linearly separable dataset with label noise
% 
% INPUTS:
%   n_points  - total number of points (evenly split between two classes)
%   dimension - dimensionality of the points
%   epsilon   - fraction of points to flip labels (0 <= epsilon <= 0.5)
%
% OUTPUTS:
%   X - (n_points x dimension) matrix of features
%   y - (n_points x 1) vector of labels (+1 or -1)

    if epsilon < 0 || epsilon > 0.5
        error('epsilon must be between 0 and 0.5');
    end

    % Generate linearly separable data
    n_per_class = n_points / 2;
    if mod(n_points, 2) ~= 0
        error('n_points must be even for balanced classes');
    end

    % Create a random hyperplane
    w = randn(dimension, 1); % weight vector
    b = randn;               % bias term

    % Generate random points
    X_pos = randn(n_per_class, dimension) + 1; % shift to one side
    X_neg = randn(n_per_class, dimension) - 1; % shift to other side

    X = [X_pos; X_neg];
    y = [ones(n_per_class, 1); -ones(n_per_class, 1)];

    % Shuffle the data
    perm = randperm(n_points);
    X = X(perm, :);
    y = y(perm);

    % Flip labels for epsilon fraction of data
    n_flip = round(n_points * epsilon);
    flip_idx = randperm(n_points, n_flip);
    y(flip_idx) = -y(flip_idx); % Flip the labels

end
