function X = prox_l12(B,lambda)

% min_X lambda*||X||_{12}+0.5*||X-B||_2^2
%column sparse

X = zeros(size(B));
for j = 1 : size(X,2)
    nxj = norm(B(:,j));
    if nxj > lambda  
        X(:,j) = (1-lambda/nxj)*B(:,j);
    end
end