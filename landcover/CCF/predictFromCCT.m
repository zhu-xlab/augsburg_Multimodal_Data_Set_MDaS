function [classLabelIds, countsLeaf] = predictFromCCT(tree,X)
%predictFromCCT predicts class using trained tree
%
% [classLabelIds, classLabels, countsLeaf] = predictFromCCT(tree,X)
%
% Inputs:   tree = output strcut from growTree
%              X = processed input features
% Outputs:  classLabelIds = Vector of numeric predictions corresponding to
%                         the class label ids.  Note this is not
%                         necessarily the class itself, e.g. if the classes
%                         are the digits 0 to 9, then classLabelId==1 is
%                         digit 0.
%           countsLeaf  = Training counts at the assigned leaf.
%
% 14/06/15

    if tree.bLeaf
        classLabelIds = tree.labelClassId*ones(size(X,1),1);
        if nargout>1
            countsLeaf = repmat(tree.trainingCounts,size(X,1),1);
        end
    else
        if isfield(tree,'rotDetails') && ~isempty(tree.rotDetails)
            X = bsxfun(@minus,X,tree.rotDetails.muX)*tree.rotDetails.R;
        end
        
        bLessChild = (X(:,tree.iIn)*tree.decisionProjection)<=tree.paritionPoint;
        
        classLabelIds = NaN(size(X,1),1);
        if nargout>1
            countsLeaf = NaN(size(X,1),numel(tree.trainingCounts));
        end
        
        if any(bLessChild)
            if nargout>1
                [classLabelIds(bLessChild), countsLeaf(bLessChild,:)] = predictFromCCT(tree.lessthanChild,X(bLessChild,:));
            else
                classLabelIds(bLessChild) = predictFromCCT(tree.lessthanChild,X(bLessChild,:));
            end
        end
        if any(~bLessChild)
            if nargout>1
                [classLabelIds(~bLessChild), countsLeaf(~bLessChild,:)] = predictFromCCT(tree.greaterthanChild,X(~bLessChild,:));
            else
                classLabelIds(~bLessChild) = predictFromCCT(tree.greaterthanChild,X(~bLessChild,:));
            end
        end
    end
end