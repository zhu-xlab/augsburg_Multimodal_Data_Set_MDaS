function R = randomRotation(N)
%randomRotation Random rotation of given dimension
    [R,~] = qr(randn(N));
    R = R/round(det(R));
end