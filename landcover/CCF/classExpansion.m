function [Y, classes] = classExpansion(Y)
%classExpansion Ensures that class data is in its binary expansion format
%
% function [Yb, classes] = classExpansion(Y)
%
% Inputs: 
%    Y       = Class information, can be a binary expansion, a numerical
%              vector of labels or a cell array of numerical or string 
%              labels
%            
% Outpus: 
%    Yb      = Input in binary expansion format
%    classes = Names of classes.  In CCT only the class index is stored and
%              so this is used to convert to the original name.
%
% Tom Rainforth 22/06/15

if size(Y,2)==1 && ~islogical(Y)
    classes = unique(Y);
    if numel(classes)==2
        % If there are only 2 classes the binary expansion can be a single
        % logical array
        YVec = Y;
        if iscell(YVec)
            Y = cellfun(@(x) strcmpi(x,classes{2}) || (x==classes{2}), YVec);
        else
            Y = YVec==classes(2);
        end
    else
        YVec = Y;
        Y = false(size(YVec,1),numel(classes));
        if iscell(YVec)
            for k=1:numel(classes)
                Y(:,k) = cellfun(@(x) strcmpi(x,classes{k}) || (x==classes{k}), YVec);
            end
        else
            for k=1:numel(classes)
                Y(:,k) = YVec==classes(k);
            end
        end
    end
else
    % Already in binary format but make sure Y is logical type to minimize 
    % memory requirement in recursion
    Y = logical(Y);
    if size(Y,2)==2
        classes = [false,true];
    else
        classes = 1:size(Y,2);
    end
end