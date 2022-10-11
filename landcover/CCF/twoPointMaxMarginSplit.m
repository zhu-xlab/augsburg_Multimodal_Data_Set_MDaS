function [bSp, rmm, cmm] = twoPointMaxMarginSplit(X,Y,tol)
% This should only be done if X has exactly 2 unique rows in which case it
% produces the optimal split
bType1 = all(abs(bsxfun(@minus,X,X(1,:)))<tol,2);
if size(Y,2)==1
    YLeft = Y;
    YRight = ~Y;
else
    YLeft = Y(bType1,:);
    YRight = Y(~bType1,:);
end
if all(sum(YLeft,1)==sum(YRight,1))
    % Here the two unique points have identical sets of class
    % labels and so we can't split
    bSp = false;
    rmm = [];
    cmm = [];
    return
else
    bSp = true;
end
% Otherwise the optimal spliting plane is the plane perpendicular
% to the vector between the two points (rmm) and the maximal
% marginal split point (cmm) is halway between the two points on
% this line.
iType2 = find(~bType1,1);
rmm = (X(iType2,:)-X(1,:))';
cmm = 0.5*(X(iType2,:)*rmm+X(1,:)*rmm);
if any(isnan(cmm(:))) || any(isinf(cmm(:)))
    error('Suggested split point at infitity / nan');
end
end