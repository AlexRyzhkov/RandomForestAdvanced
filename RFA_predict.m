function Y_pred = RFA_predict(RFA, Xtest)
N_class = RFA.N_class;
ntrees = RFA.nTrees;
Y_pred = zeros(size(Xtest, 1), N_class);

for i = 1:ntrees
    Y_pred = Y_pred + RFA_treePredict(RFA.Trees{i}, Xtest, N_class);
end

Y_pred = Y_pred / ntrees;
end
