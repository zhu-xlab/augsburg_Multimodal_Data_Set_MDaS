function bLessThanTwoUniqueRows = queryIfOnlyTwoUniqueRows(X)
% Function that checks if an array has only two unique rows as this can
% cause failure of for example LDA
if size(X,1)==2
    bLessThanTwoUniqueRows = true;
    return
end
bEqualFirst = all(bsxfun(@eq,X,X(1,:)),2);
iFirstNotEqual = find(~bEqualFirst,1);
if isempty(iFirstNotEqual)
    bLessThanTwoUniqueRows = true;
    return;
end
iToCheck = find(~bEqualFirst(2:end))+1;
bNotUnique = all(bsxfun(@eq,X(iToCheck,:),X(iFirstNotEqual(1),:)),2);
bLessThanTwoUniqueRows = all(bNotUnique);
end