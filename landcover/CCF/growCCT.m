function tree = growCCT(XTrain,YTrain,options,iFeatureNum,depth)
%growTree grows a CCT, recursively calling itself until leaves are reached
%
% function CCT = growCCT(XTrain,YTrain,options,iFeatureNum,depth)
%
% Function applies greedy splitting according to the CCT algorithm and the
% provided options structure.  This is equivalent to alogrithm 2 in the
% paper if the options structure is default.  Algorithm either returns a
% leaf or forms an internal splitting node in which case the function
% recursively calls itself for each of the children.
%
% Inputs:
%    XTrain      = Array giving training features.  Data should be
%                  processed using processInputData before being passed to
%                  CCT
%    YTrain      = Class data formatted as per output of classExpansion
%    options     = Options class of type optionsClassCCT.  Some fields are
%                  updated during recursion
%    iFeatureNum = Grouping of features as per processInputData.  During
%                  recursion if a feature is found to be identical across
%                  data points, the corresponding values in iFeatureNum are
%                  replaced with NaNs.
%    depth       = Current tree depth (zero based)
%
% Outputs
%   tree         = Structure containing learnt tree.  Prediction can be
%                  made using predictFromCCT
%
% Tom Rainforth 22/06/15

%% First do checks for whether we should immediately terminate

N = size(XTrain,1);
% Return if one training point, pure node or if options for returning
% fulfilled.  A little case to deal with a binary YTrain is required.
bStop = (N<(max(2,options.minPointsForSplit))) || (isnumeric(options.maxDepthSplit) && depth>options.maxDepthSplit);
if bStop
    tree = setupLeaf(YTrain,options);
    return
elseif size(YTrain,2)>1
    if (sum(abs(sum(YTrain,1))>1e-12)<2)
        tree = setupLeaf(YTrain,options);
        return
    end
elseif any(sum(YTrain)==[0,size(YTrain,1)])
    tree = setupLeaf(YTrain,options);
    return
end

if depth>490 && strcmpi(options.maxDepthSplit,'stack')
    error('Tree is too deep and causing stack issues');
end

%% Subsample features as required for hyperplane sampling

iCanBeSelected = fastUnique(iFeatureNum);
iCanBeSelected(isnan(iCanBeSelected))=[];
lambda = min(numel(iCanBeSelected),options.lambda);
indFeatIn = randperm(numel(iCanBeSelected),lambda);
iFeatIn = iCanBeSelected(indFeatIn);

bInMat = bsxfun(@eq,iFeatureNum(:)',sort(iFeatIn(:)));

iIn = find(any(bInMat,1));


% Check for variation along selected dimensions and resample features that
% have no variation
bXVaries = queryIfColumnsVary(XTrain(:,iIn),options.XVariationTol);

if ~all(bXVaries)
    iInNew = iIn;
    nSelected = 0;
    iIn = iIn(bXVaries);
    while ~all(bXVaries) && lambda>0
        iFeatureNum(iInNew(~bXVaries)) = NaN;
        bInMat(:,iInNew(~bXVaries)) = false;
        bRemainsSelected = any(bInMat,2);
        nSelected = nSelected+sum(bRemainsSelected);
        iCanBeSelected(indFeatIn) = [];
        lambda = min(numel(iCanBeSelected),options.lambda-nSelected);
        if lambda<1
            break
        end
        indFeatIn = randperm(numel(iCanBeSelected),lambda);
        iFeatIn = iCanBeSelected(indFeatIn);
        bInMat = bsxfun(@eq,iFeatureNum(:)',iFeatIn(:));
        iInNew = find(any(bInMat,1));
        bXVaries = queryIfColumnsVary(XTrain(:,iInNew),options.XVariationTol);
        iIn = sort([iIn,iInNew(bXVaries)]);
    end
end

if isempty(iIn)
    % This means that there was no variation along any feature, therefore
    % exit.
    tree = setupLeaf(YTrain,options);
    return
end


%% Projection bootstrap if required

if options.bProjBoot
    iTrainThis = randi(size(XTrain,1),size(XTrain,1),1);
    XTrainBag = XTrain(iTrainThis,iIn);
    YTrainBag = YTrain(iTrainThis,:);
else
    XTrainBag = XTrain(:,iIn);
    YTrainBag = YTrain;
end

bXBagVaries = queryIfColumnsVary(XTrainBag,options.XVariationTol);
if ~any(bXBagVaries) || (size(YTrainBag,2)>1 && (sum(abs(sum(YTrainBag,1))>1e-12)<2)) || (size(YTrainBag,2)==1 && any(sum(YTrainBag)==[0,size(YTrainBag,1)]))
    if ~options.bContinueProjBootDegenerate
        tree = setupLeaf(YTrain,options);
        return
    else
        XTrainBag = XTrain(:,iIn);
        YTrainBag = YTrain;
    end
end


%% Check for only having two points

if ~isempty(options.projections) && ((size(XTrainBag,1)==2) || queryIfOnlyTwoUniqueRows(XTrainBag))
    % If there are only two points setup a maximum marginal split between the points
    
    [bSplit,projMat,partitionPoint] = twoPointMaxMarginSplit(XTrainBag,YTrainBag,options.XVariationTol);
    if ~bSplit
        tree = setupLeaf(YTrain,options);
        return
    else
        bLessThanTrain = (XTrain(:,iIn)*projMat)<=partitionPoint;
        iDir = 1;
    end
else
    % Generate the new features as required
    
    if ~isempty(options.projections)
        projMat = componentAnalysis(XTrainBag,YTrainBag,options.projections,options.epsilonCCA);
    end
    
    %% Choose the features to use
    
    if ~ischar(options.includeOriginalAxes) && ~options.includeOriginalAxes
        if isempty(projMat)
            error('Must make new features to have includeOriginalAxes false');
        end
    elseif strcmpi(options.includeOriginalAxes,'sampled')
        projMat = [projMat,eye(size(projMat,1))];
    elseif strcmpi(options.includeOriginalAxes,'all')
        projMatNew = zeros(size(XTrain,2),size(projMat,2));
        projMatNew(iIn,iIn) = projMat;
        iIn = find(~isnan(iFeatureNum));
        projMat = [projMatNew(iIn,iIn),eye(numel(iIn))];
    else
        error('Invalid option for includeOriginalAxes');
    end
    
    UTrain = XTrain(:,iIn)*projMat;
    % This step catches splits based on no significant variation
    bUTrainVaries = queryIfColumnsVary(UTrain,options.XVariationTol);
    
    if ~any(bUTrainVaries)
        tree = setupLeaf(YTrain,options);
        return
    end
    
    UTrain = UTrain(:,bUTrainVaries);
    projMat = projMat(:,bUTrainVaries);
    
    %% Search over splits using provided method
    
    nProjDirs = size(UTrain,2);
    splitGains = NaN(nProjDirs,1);
    iSplits = NaN(nProjDirs,1);
    
    for nVarAtt = 1:nProjDirs
        
        % Calculate the probabilities of being at each class in each of child
        % nodes based on proportion of training data for each of possible
        % splits using current projection
        [UTrainSort,iUTrainSort] = sort(UTrain(:,nVarAtt));
        YTrainSort = YTrain(iUTrainSort,:);
        if size(YTrain,2)==1
            LeftCumCounts = [(1:numel(YTrainSort))'-cumsum(YTrainSort),cumsum(YTrainSort)];
        else
            LeftCumCounts = cumsum(YTrainSort,1);
        end
        RightCumCounts = bsxfun(@minus,LeftCumCounts(end,:),LeftCumCounts);
        bUniquePoints = [diff(UTrainSort,[],1)>options.XVariationTol;false];
        pL = bsxfun(@rdivide,LeftCumCounts,sum(LeftCumCounts,2));
        pR = bsxfun(@rdivide,RightCumCounts,sum(RightCumCounts,2));
        
        % Calculate the metric values of the current node and two child nodes
        if strcmpi(options.splitCriterion,'gini')
            metricLeft = 1-sum(pL.^2,2);
            metricRight = 1-sum(pR.^2,2);
        elseif strcmpi(options.splitCriterion,'info')
            pLProd = pL.*log2(pL);
            pLProd(pL==0) = 0;
            metricLeft = -sum(pLProd,2);
            pRProd = pR.*log2(pR);
            pRProd(pR==0) = 0;
            metricRight = -sum(pRProd,2);
        else
            error('Invalid split criterion');
        end
        metricCurrent = metricLeft(end);
        metricLeft(~bUniquePoints) = inf;
        metricRight(~bUniquePoints) = inf;
        
        % Calculate gain in metric for each of possible splits based on current
        % metric value minus metric value of child weighted by number of terms
        % in each child
        metricGain = metricCurrent-((1:N)'.*metricLeft+(N-1:-1:0)'.*metricRight)/N;
        
        % Randomly sample from equally best splits
        [splitGains(nVarAtt),iSplits(nVarAtt)] = max(metricGain(1:end-1));
        iEqualMax = find(abs(metricGain(1:end-1)-splitGains(nVarAtt))<(10*eps));
        if isempty(iEqualMax)
            iEqualMax = 1;
        end
        iSplits(nVarAtt) = iEqualMax(randi(numel(iEqualMax)));
        
    end
    
    % If no split gives a positive gain then stop
    if max(splitGains)<0
        tree = setupLeaf(YTrain,options);
        return
    end
    
    % Establish between projection direction
    maxGain = max(splitGains);
    iEqualMax = find(abs(splitGains-maxGain)<(10*eps));
    % Use given method to break ties
    if strcmpi(options.dirIfEqual,'rand')
        iDir = iEqualMax(randi(numel(iEqualMax)));
    elseif strcmpi(options.dirIfEqual,'first')
        iDir = iEqualMax(1);
    else
        error('invalid dirIfEqual');
    end
    iSplit = iSplits(iDir);
    
    
    %% Establish partition point and assign to child
    
    UTrain = UTrain(:,iDir);
    UTrainSort = sort(UTrain);
    
    % The convoluted nature of the below is to avoid numerical errors
    uTrainSortLeftPart = UTrainSort(iSplit);
    UTrainSort = UTrainSort-uTrainSortLeftPart;
    partitionPoint = UTrainSort(iSplit)*0.5+UTrainSort(iSplit+1)*0.5;
    partitionPoint = partitionPoint+uTrainSortLeftPart;
    UTrainSort = UTrainSort+uTrainSortLeftPart; %#ok<NASGU>
    
    bLessThanTrain = UTrain<=partitionPoint;
    
    if ~any(bLessThanTrain) || all(bLessThanTrain)
        error('Suggested split with empty');
    end
    
end

%% Recur tree growth to child nodes and constructs tree struct
% to return

% Update ancestral counts for breaking ties if needed
if size(YTrain,2)==1
    countsNode = [numel(YTrain)-sum(YTrain),sum(YTrain)];
else
    countsNode = sum(YTrain,1);
end
nNonZeroCounts = sum(countsNode>0);
nUniqueNonZeroCounts = numel(fastUnique(countsNode));
if nUniqueNonZeroCounts==nNonZeroCounts
    options.ancestralProbs = countsNode/sum(countsNode);
else
    options.ancestralProbs = [options.ancestralProbs;countsNode/sum(countsNode)];
end

treeLeft = growCCT(XTrain(bLessThanTrain,:),YTrain(bLessThanTrain,:),options,iFeatureNum,depth+1);
treeRight = growCCT(XTrain(~bLessThanTrain,:),YTrain(~bLessThanTrain,:),options,iFeatureNum,depth+1);
tree.bLeaf = false;
tree.trainingCounts = countsNode;
tree.iIn = iIn;
tree.decisionProjection = projMat(:,iDir);
tree.paritionPoint = partitionPoint;
tree.lessthanChild = treeLeft;
tree.greaterthanChild = treeRight;

end

function tree = setupLeaf(YTrain,options)
% Update tree struct to make node a leaf

if size(YTrain,2)==1
    countsNode = [numel(YTrain)-sum(YTrain),sum(YTrain)];
else
    countsNode = sum(YTrain,1);
end
maxCounts = max(countsNode);
bEqualMaxCounts = maxCounts == countsNode;
if sum(bEqualMaxCounts)==1
    label = find(bEqualMaxCounts);
else
    nRecur = size(options.ancestralProbs,1);
    while nRecur>0
        countsTieBreak = countsNode+options.ancestralProbs(nRecur,:)/1e9;
        maxCounts = max(countsTieBreak);
        bEqualMaxCounts = maxCounts == countsTieBreak;
        if sum(bEqualMaxCounts)==1
            label = find(bEqualMaxCounts);
            break
        else
            nRecur = nRecur-1;
        end
        if nRecur==0
            [~,label] = max(countsNode+rand(size(countsNode))/1e9);
        end
    end
end
tree.bLeaf = true;
tree.labelClassId = label;
tree.trainingCounts = countsNode;
end
% 
% function [bSp, rmm, cmm] = twoPointMaxMarginSplit(X,Y,tol)
% % This should only be done if X has exactly 2 unique rows
% bType1 = all(abs(bsxfun(@minus,X,X(1,:)))<tol,2);
% if size(Y,2)==1
%     YLeft = Y;
%     YRight = ~Y;
% else
%     YLeft = Y(bType1,:);
%     YRight = Y(~bType1,:);
% end
% if all(sum(YLeft,1)==sum(YRight,1))
%     % Here the two unique points have identical sets of class
%     % labels and so we can't split
%     bSp = false;
%     rmm = [];
%     cmm = [];
%     return
% else
%     bSp = true;
% end
% % Otherwise the optimal spliting plane is the plane perpendicular
% % to the vector between the two points (rmm) and the maximal
% % marginal split point (cmm) is halway between the two points on
% % this line.
% iType2 = find(~bType1,1);
% rmm = (X(iType2,:)-X(1,:))';
% cmm = 0.5*(X(iType2,:)*rmm+X(1,:)*rmm);
% if isnan(cmm) || isinf(cmm)
%     error('Suggested split point at infitity / nan');
% end
% end
% 
% function bVar = queryIfColumnsVary(XvarToTest,tol)
% % Function that says whether columns are constant or not
% bVar = max(abs(diff(XvarToTest(1:min(5,size(XvarToTest,1)),:),[],1)),[],1)>tol;
% bVar(~bVar) = max(abs(diff(XvarToTest(:,~bVar),[],1)),[],1)>tol;
% end
% 
% function bLessThanTwoUniqueRows = queryIfOnlyTwoUniqueRows(X)
% % Function that checks if an array has only two unique rows as this can
% % cause failure of for example LDA
% if size(X,1)==2
%     bLessThanTwoUniqueRows = true;
%     return
% end
% bEqualFirst = all(bsxfun(@eq,X,X(1,:)),2);
% iFirstNotEqual = find(~bEqualFirst,1);
% if isempty(iFirstNotEqual)
%     bLessThanTwoUniqueRows = true;
%     return;
% end
% iToCheck = find(~bEqualFirst(2:end))+1;
% bNotUnique = all(bsxfun(@eq,X(iToCheck,:),X(iFirstNotEqual,:)),2);
% bLessThanTwoUniqueRows = all(bNotUnique);
% end