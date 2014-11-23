function Tree = RFA_trainTree(Xtrain, Ytrain, options)
N_class = options.N_class;
Tree = zeros(100000, 4 + N_class);
[Tree, cnt_new] = RFA_constructTree(1, Tree, Xtrain, Ytrain, options);
Tree(cnt_new:end, :) = [];
end