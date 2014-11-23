% load fisheriris
% N = size(meas, 1);
% X = meas;
% Y = zeros(1, N);
% Y(strcmp(species, 'versicolor')) = 1;
% Y(strcmp(species, 'virginica')) = 2;
% N = 150;

% load ionosphere
% N = size(X, 1);
% Y_data = Y;
% Y = zeros(1, N);
% Y(strcmp(Y_data, 'b')) = 0;
% Y(strcmp(Y_data, 'g')) = 1;

load Boson_little
Y = Y';
X = X(1:2500, :);
Y= Y(1:2500);
N = size(X, 1);

options = struct('nTrees', 10, 'maxLeafSize', 10, 'nBords', 7000);

Results = zeros(1, 1);
L = round(0.8 * N);
for i = 1: 10
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
