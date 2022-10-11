function [R, muX, U] = rotationForestDataProcess(X,M,pS)

error('Update so as to always ignore some class');

if ~exist('pS','var') || isempty(pS)
    pS = 0.5;
end

muX = mean(X,1);
X = bsxfun(@minus,X,muX);

D = size(X,2);

fOrder = randperm(D);

fGroups = reshape(fOrder(1:(M*floor(D/M))),M,[]);
fLeft = fOrder((M*floor(D/M)+1):end);

R = sparse(D,D);
iUpTo = 1;

K = size(fGroups,2);

for n=1:K    
    r = localRotation(X(:,fGroups(:,n)),pS);
    R((1+(n-1)*M):(n*M),iUpTo:(iUpTo+size(r,2)-1)) = r;
    iUpTo = iUpTo+size(r,2);
end

if ~isempty(fLeft)
    r = localRotation(X(:,fLeft),pS);
    R((1+(K)*M):end,iUpTo:(iUpTo+size(r,2)-1)) = r;
    iUpTo = iUpTo+size(r,2);
end

R = R(:,1:(iUpTo-1));

R(fOrder,:) = R;

if nargout>2
    U = X*R;
end

end

function r = localRotation(x,p)
    
    iB = datasample((1:size(x,1)),round(size(x,1)*p));
    xB = x(iB,:);
    r = pcaReduced(xB,false,true);

end