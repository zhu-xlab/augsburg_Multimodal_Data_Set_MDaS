function bVar = queryIfColumnsVary(XvarToTest,tol)
% Function that says whether columns are constant or not
bVar = max(abs(diff(XvarToTest(1:min(5,size(XvarToTest,1)),:),[],1)),[],1)>tol;
bVar(~bVar) = max(abs(diff(XvarToTest(:,~bVar),[],1)),[],1)>tol;
end