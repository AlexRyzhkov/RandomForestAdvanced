function [Tree, cnt_new] = RFA_constructTree(cnt, Tree, Xtrain, Ytrain, options)
maxLeafSize = options.maxLeafSize;
N_class = options.N_class;
counts = zeros(N_class, 1);
for i = 1 : N_class
    counts(i) = sum(Ytrain == i - 1);
end
if (size(Xtrain, 1) <= maxLeafSize || sum(counts == length(Ytrain)) > 0)
    for i = 0 : N_class - 1
        Tree(cnt, 5 + i) = counts(i + 1) / length(Ytrain);
    end
    cnt_new = cnt + 1;
    return
end
N = size(Xtrain, 2);
while (true)
    feat = randi([1 N]);
    bord = RFA_chooseBord(Xtrain(:, feat), Ytrain, options);
    Tree(cnt, 1) = feat;
    Tree(cnt, 2) = bord;
    Tree(cnt, 3) = cnt + 1;
    Mask = (Xtrain(:, feat) < bord);
    if sum(Mask) ~= 0 && sum(Mask) ~= length(Mask)
        break
    end
end
[Tree, cnt_new] = RFA_constructTree(cnt + 1, Tree, Xtrain(Mask, :), Ytrain(Mask), options);
Tree(cnt, 4) = cnt_new;
[Tree, cnt_new] = RFA_constructTree(cnt_new, Tree, Xtrain(~Mask, :), Ytrain(~Mask), options);
end