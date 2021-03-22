function Out = QualityIndices(I_HS,I_REF,ratio)
%--------------------------------------------------------------------------
% Quality Indices
%
% USAGE
%   Out = QualityIndices(I_HS,I_REF,ratio)
%
% INPUT
%   I_HS    : target HS data (rows,cols,bands)
%   I_REF   : reference HS data (rows,cols,bands)
%   ratio   : GSD ratio between HS and MS imagers
%
% OUTPUT
%   Out.psnr : PSNR
%   Out.sam  : SAM
%   Out.ergas: ERGAS
%   Out.q2n  : Q2N
%
%--------------------------------------------------------------------------


[angle_SAM,map] = SAM(I_HS,I_REF);
Out.sam = angle_SAM;
Out.ergas = ERGAS(I_HS,I_REF,ratio);
psnr = PSNR(I_REF,I_HS);
Out.psnrall = psnr.all;
Out.sammap = map;
Out.psnr = psnr.ave;
[Q2n_index, ~] = q2n(I_REF, I_HS, 32, 32);
Out.q2n = Q2n_index;

disp(['PSNR : ' num2str(Out.psnr)]);
disp(['SAM  : ' num2str(Out.sam)]);
disp(['ERGAS: ' num2str(Out.ergas)]);
disp(['Q2n  : ' num2str(Out.q2n)]);