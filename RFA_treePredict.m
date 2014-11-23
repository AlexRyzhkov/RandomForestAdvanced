function Res = RFA_treePredict(Tree, Xtest, N_class)
N = size(Xtest, 1);
Res = zeros(N, N_class);
for i = 1:N
    X = Xtest(i, :);
    cnt = 1;
    while true
        if Tree(cnt, 1) == 0
            Res(i, :) = Tree(cnt, 5 : 4 + N_class);
            break;
        end
        if X(Tree(cnt, 1)) < Tree(cnt, 2)
            cnt = Tree(cnt, 3);
        else
            cnt = Tree(cnt, 4); 
        end
    end
end

end
