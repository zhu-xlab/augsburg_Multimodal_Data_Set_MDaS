function ERGAS = ERGAS(I,I_Fus,Resize_fact,mask)
%--------------------------------------------------------------------------
% Erreur relative globale adimensionnelle de synthese (ERGAS)
%
% USAGE
%   ERGAS = ERGAS(I,I_Fus,Resize_fact,mask)
%
% INPUT
%   I          : reference HS data (rows,cols,bands)
%   I_Fus      : target HS data (rows,cols,bands)
%   Resize_fact: resize factor
%   mask       : binary mask (rows,cols) (optional)
%
% OUTPUT
%   ERGAS : ERGAS (scalar)
%
%--------------------------------------------------------------------------
if nargin == 3
    I = double(I);
    I_Fus = double(I_Fus);

    Err=I-I_Fus;
    ERGAS=0;
    for iLR=1:size(Err,3),
        ERGAS=ERGAS+mean2(Err(:,:,iLR).^2)/(mean2((I(:,:,iLR))))^2;   
    end

    ERGAS = (100/Resize_fact) * sqrt((1/size(Err,3)) * ERGAS);
else
    I = double(I);
    I_Fus = double(I_Fus);

    Err=I-I_Fus;
    ERGAS=0;
    for iLR=1:size(Err,3),
        tmp1 = Err(:,:,iLR);
        tmp2 = I(:,:,iLR);
        ERGAS=ERGAS+mean(tmp1(mask~=0).^2)/(mean(tmp2(mask~=0)))^2;   
    end

    ERGAS = (100/Resize_fact) * sqrt((1/size(Err,3)) * ERGAS);

end