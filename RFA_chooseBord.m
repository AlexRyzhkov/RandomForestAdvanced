function bord = RFA_chooseBord(Xtrain, Ytrain, options)
%nBords = options.nBords;
N = size(Xtrain, 1);
N_class = options.N_class;

% ----------------------------------------------
% mn = min(Xtrain);
% mx = max(Xtrain);
% 
% Bords = (mx-mn) * rand(1, nBords) + mn;
% Mask = (repmat(Xtrain, 1, nBords) < repmat(Bords, N, 1));
% ----------------------------------------------



[X, ind] = sort(Xtrain);
Y = Ytrain(ind);

dfs = logical(abs(diff(Y)));
S = (X(1:end-1) + X(2:end)) / 2;
Bords = S(dfs)';
nBords = length(Bords);
Mask = (repmat(Xtrain, 1, nBords) < repmat(Bords, N, 1));

IG = IGain(Ytrain, Mask', N_class);
[~, ind] = max(IG);

bord = Bords(ind);

end