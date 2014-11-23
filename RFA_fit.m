function RFA = RFA_fit(Xtrain, Ytrain, options)
N = length(Ytrain);
nTrees = options.nTrees;
Trees = cell(nTrees, 1);
N_class = length(unique(Ytrain));
options.N_class = N_class;

for i = 1:nTrees
    q = randi([1 N], N, 1);
    X = Xtrain(q, :);
    Y = Ytrain(q);
    Trees(i) = {RFA_trainTree(X, Y, options)};
    %RFA_plotTree(Trees{i});
end
RFA = struct('Trees', {Trees}, 'nTrees', nTrees, 'N_class', N_class);
end