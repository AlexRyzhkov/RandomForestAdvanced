function IG = IGain(Y, Mask, N_class)
Y = Y(:)' + 1;
[M, N] = size(Mask);

N_left = sum((1 - Mask), 2);
N_right = N - N_left;
N_left_cor = N_left + (N_left == 0);
N_right_cor = N_right + (N_right == 0);

Y = repmat(Y, M, 1);
Y1 = Y .* (1 - Mask);
Y2 = Y .* Mask;

E_all = zeros(M, 1);
E_left = zeros(M, 1);
E_right = zeros(M, 1);

for i = 1:N_class
    S = sum(Y == i, 2) / N;
    S(S == 0) = 1;
    E_all = E_all - S .* log2(S);
    
    S = sum(Y1 == i, 2) ./ N_left_cor;
    S(S == 0) = 1;
    E_left = E_left - S .* log2(S);
    
    S = sum(Y2 == i, 2) ./ N_right_cor;
    S(S == 0) = 1;
    E_right = E_right - S .* log2(S);
end

IG = E_all - (N_left .* E_left + N_right .* E_right) / N;
end