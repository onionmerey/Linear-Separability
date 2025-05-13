data=Generate_LS_data(50,10000)'; % Adjust filename/path as needed

X = data(:, 2:end)';
y = data(:, 1)';

y(y == 0) = -1;


[d, n] = size(X);

tic

A = zeros(n, d+1);
for i = 1:n
    A(i, :) = -y(i) * [X(:, i)' 1];
end
b_vec = -ones(n, 1);

f = zeros(d+1, 1);

options = optimoptions('linprog', 'Display', 'none', 'Algorithm','interior-point');

[v, ~, exitflag, ~] = linprog(f, A, b_vec, [], [], [], [], options);


if exitflag == 1
    disp('The data is linearly separable.');
    w = v(1:d);
    b_param = v(end);
    %fprintf('Hyperplane parameters: w = [%s], b = %f\n', num2str(w.'), b_param);
else
    disp('The data is not linearly separable.');
end

toc
