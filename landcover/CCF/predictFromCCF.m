function [forestPredicts, forestProbs, treePredictions, cumulativeForestPredicts] = predictFromCCF(CCF,X)
%predictFromCCF predicts class using trained forest
%
% [prediction, countsLeaf] = predictFromCCF(CCF,X)
%
% Inputs:                 CCF = output from genCCF.  This is a structure
%                               with a field Trees, giving a cell array of
%                               tree structures, and options which is an
%                               object of type optionsClassCCT
%                           X = input features, each row should be a 
%                               seperate data point
% Outputs:  forestPredictions = Vector of numeric predictions corresponding to
%                               the class label
%                 forestProbs = Assigned probability to each class
%             treePredictions = Individual tree predictions
%    cumulativeForestPredicts = Predictions of forest cumaltive with adding
%                               trees
%
% 14/06/15

X = CCF.inputProcess(X);

nTrees = numel(CCF.Trees);
treePredictions = NaN(size(X,1),nTrees);

for n=1:nTrees
    treePredictions(:,n) = predictFromCCT(CCF.Trees{n},X);
end

K = numel(CCF.Trees{1}.trainingCounts);

if nargout>3
   cumVotes = bsxfun(@rdivide,cumsum(bsxfun(@eq,treePredictions,reshape(1:K,[1,1,K])),2),reshape(1:nTrees,[1,nTrees,1]));
   forestProbs = squeeze(cumVotes(:,end,:));
   voteFactor = reshape(CCF.options.voteFactor/mean(CCF.options.voteFactor),[1,1,K]);
   [~,cumulativeForestPredicts] = max(bsxfun(@times,cumVotes,voteFactor),[],3);
   forestPredicts = cumulativeForestPredicts(:,end);
else
   forestProbs = squeeze(sum(bsxfun(@eq,treePredictions,reshape(1:K,[1,1,K])),2))/nTrees;
   [~,forestPredicts] = max(bsxfun(@times,forestProbs,CCF.options.voteFactor(:)'),[],2);
end

end