function aucScore = auroc(tp, fp)
    n = size(tp, 1);
    aucScore = sum((fp(2:n) - fp(1:n-1)).*(tp(2:n)+tp(1:n-1)))/2;
end