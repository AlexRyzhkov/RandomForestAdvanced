N = 20000;
M = 50;
X = rand(N, M);
Y = round(rand(1, N));

options = struct('nTrees', 10, 'maxLeafSize', 50, 'nBords', 7000);

Results = zeros(1, 1);
L = round(0.8 * N);
for i = 1: 1
    i
    q = randperm(N);
    XTrain = X(q(1:L), :);
    XTest = X(q(L + 1:N), :);
    YTrain = Y(q(1:L));
    YTest = Y(q(L + 1:N));
    
    RFA = RFA_fit(XTrain, YTrain, options);
    Y_pred = RFA_predict(RFA, XTest);
    [~, Y_ind] = max(Y_pred, [], 2);
    Y_ind = Y_ind - 1;
    
    Results(i) = sum((YTest' - Y_ind) .^ 2) / length(Y_ind);
    fprintf('Result_i = %.5f\n', Results(i))
end

fprintf('Error = %.5f\n', mean(Results))
