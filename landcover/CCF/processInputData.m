function [XTrain, iFeatureNum, inputProcess, XTest, featureNames] = processInputData(XTrainRC,bOrdinal,XTestRC)
%processInputData Process input features, expanding categoricals and
%converting to zScores
%
% function [XTrain, iFeatureNum, inputProcess, XTest, featureNames] ...
%                       = processInputData(XTrain,bOrdinal,XTest)
%
% Required Inputs
%   XTrain   = Unprocessed input features, can be a numerical array, a cell
%              array or a table.  Each row is a seperate data point.
% Options Inputs
%   bOrdinal = Logical array indicating if the corresponding feature is
%              ordinal (i.e. numerical or order categorical).  The default
%              treatment is that numerical inputs are ordinal and strings
%              are not.  If a feature contains for numerical features and
%              strings, it is presumed to not be ordinal unless there is
%              only a single string category, in which case this is
%              presumed to indicate missing data.
%   XTest    = Additional data to be transformed.  This is seperate to the
%              training data for the purpose of Z-scores and to avoid using
%              any features / categories that do not appear in the training
%              data.
% Outputs
%  XTrain        = Processed input features
%  iFeatureNum   = Array idenitifying groups of expanded features.
%                  Expanded features with the same iFeatureNum value come 
%                  from the same original non-ordinal feature.
%  inputProcess  = Anonymous function that can be used for converting
%                  new data requiring only the new data as an input.
%  XTest         = Additional processed input features
%  featureNames  = Names of the expanded features.  Variable names are
%                  taken from the table properties if in the input is a
%                  cell.  For expanded categorical features the name of the
%                  category is also included.
%
% Tom Rainforth 22/06/15

D = size(XTrainRC,2);

if istable(XTrainRC)
    % Keep names if given
    featureNamesOrig = XTrainRC.Properties.VariableNames;
    try
        XTrainRC = table2array(XTrainRC);
    catch
        XTrainRC = table2cell(XTrainRC);
    end
else
    featureNamesOrig = cellfun(@(x) ['Var' num2str(x)],num2cell(1:D),'UniformOutput',false);
end

if ~exist('bOrdinal','var') || isempty(bOrdinal)
    if ~iscell(XTrainRC)
        % Default is that if input is all numeric, everything is treated as
        % ordinal
        bOrdinal = true(1,D);
    else
        % Numeric features treated as ordinal, features with only a single
        % unqiue string and otherwise numeric treated also treated as 
        % ordinal with the string taken to give a missing value and 
        % features with more than one unique string taken as non-ordinal
        bNumeric = cellfun(@isnumeric,XTrainRC);
        iContainsString = find(sum(~bNumeric,1)>0);
        nStr = zeros(1,size(XTrainRC,2));
        for n=iContainsString(:)'
            nStr(n) = numel(unique(XTrainRC(~bNumeric(:,n),n)));
        end
        bOrdinal = nStr<2;
        iSingleStr = find(nStr==1);
        for n=1:numel(iSingleStr)
            XTrainRC(~bNumeric(:,iSingleStr(n)),iSingleStr(n)) = {NaN};
        end
    end
elseif numel(bOrdinal)~=size(XTrainRC,2)
    error('bOrdinal must match size of XTrainRC');
end

XTrain = XTrainRC(:,bOrdinal);
if iscell(XTrain)
    % Anything not numeric in the ordinal features taken to be missing
    % values
    bNumeric = cellfun(@isnumeric,XTrain);
    Xrep = cellfun(@(x) sscanf(x,'%f'), XTrain(~bNumeric),'UniformOutput',false);
    Xrep(cellfun(@isempty,Xrep)) = {NaN};
    XTrain(~bNumeric) = Xrep;
    XTrain = cell2mat(XTrain);
end
XCat = XTrainRC(:,~bOrdinal);
if ~iscell(XCat)
    XCat = num2cell(XCat);
end
XCat = makeSureString(XCat,10);

iFeatureNum = 1:size(XTrain,2);
featureNames = featureNamesOrig(bOrdinal);

featureBaseNames = featureNamesOrig(~bOrdinal);

Cats = cell(size(XCat,2),1);

% Expand the categorical features
for n=1:size(XCat,2)
    Cats{n} = unique(XCat(:,n))';
    newNames = cellfun(@(x) [featureBaseNames{n}, 'Cat', x], Cats{n}, 'UniformOutput', false);
    featureNames = [featureNames, newNames]; %#ok<AGROW>
    nCats = numel(Cats{n});
    if nCats==1
        % This is setup so that any trivial features are not included
        continue
    end
    sizeSoFar = numel(iFeatureNum);
    if isempty(iFeatureNum)
        iFeatureNum = ones(1,nCats);
    else
        iFeatureNum = [iFeatureNum,(iFeatureNum(end)+1)*ones(1,nCats)]; %#ok<AGROW>
    end
    XTrain = [XTrain,zeros(size(XTrain,1),nCats)]; %#ok<AGROW>
    for c=1:nCats
        XTrain(strcmp(XCat(:,n),Cats{n}{c}),(sizeSoFar+c)) = 1;
    end
end

% Convert to Z-scores
mu_XTrain = nanmean(XTrain,1);
std_XTrain = nanstd(XTrain,[],1);
std_XTrain(abs(std_XTrain)<1e-10) = 1;
XTrain = bsxfun(@rdivide,bsxfun(@minus,XTrain,mu_XTrain),std_XTrain);
XTrain(isnan(XTrain)) = 0;

% If required, generate function for converting additional data and
% calculate conversion for any test data provided.
if nargout>2
    inputProcess = createFutureConverterFunc(bOrdinal,Cats,mu_XTrain,std_XTrain);
    if nargout>3
        XTest = inputProcess(XTestRC);
    end
end
    
end

function A = makeSureString(A,nSigFigTol)
% Ensure that all numerical values are strings (noting that the

bNum = isnumeric(A);
A(bNum) = num2str(A(bNum),nSigFigTol);
end

function f = createFutureConverterFunc(bOrdinal,Cats,mu_XTrain,std_XTrain)
    f = @(Xrc) futureConvert(Xrc,bOrdinal,Cats,mu_XTrain,std_XTrain);
end

function X = futureConvert(Xrc,bOrdinal,Cats,mu_XTrain,std_XTrain)
% This can be used to create an anonymous function that applies the same
% data transformation as was done by on the training data to new data

if size(Xrc,2)~=numel(bOrdinal)
    error('Incorrect number of features');
end

if istable(Xrc)
    try
        Xrc = table2array(Xrc);
    catch
        Xrc = table2cell(Xrc);
    end
end

X = Xrc(:,bOrdinal);
if iscell(X)
    bNumeric = cellfun(@isnumeric,X);
    Xrep = cellfun(@(x) sscanf(x,'%f'), X(~bNumeric),'UniformOutput',false);
    Xrep(cellfun(@isempty,Xrep)) = {NaN};
    X(~bNumeric) = Xrep;
    X = cell2mat(X);
end

XCat = Xrc(:,~bOrdinal);
if ~iscell(XCat)
    XCat = num2cell(XCat);
end
XCat = makeSureString(XCat,10);

for n=1:size(XCat,2)
    nCats = numel(Cats{n});
    if nCats==1
        continue
    end
    sizeSoFar = size(X,2);
    X = [X,zeros(size(X,1),nCats)]; %#ok<AGROW>
    for c=1:nCats
        X(strcmp(XCat(:,n),Cats{n}{c}),(sizeSoFar+c)) = 1;
    end
end

X = bsxfun(@rdivide,bsxfun(@minus,X,mu_XTrain),std_XTrain);
X(isnan(X)) = 0;

end