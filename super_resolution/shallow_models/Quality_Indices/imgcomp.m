function imgcomp(hyp,fused,rgb_bands)

%--------------------------------------------------------------------------
% Compare rgb composite images of hyperspectral and fused data
% 
% USAGE
%       imgcomp(hyp,fused,rgb_bands)
%
% INPUT
%   hyp      : hs data    (rows1,cols1,bands1)
%   fused    : fused data (rows2,cols2,bands2)
%   rgb_bands: bands for rgb (ex. [30, 20, 10])
%--------------------------------------------------------------------------

hyp_rgb = zeros(size(hyp,1),size(hyp,2),3);
fused_rgb = zeros(size(fused,1),size(fused,2),3);
for b = 1:3
    tmp_hyp = hyp(:,:,rgb_bands(b));
    tmp_fused = fused(:,:,rgb_bands(b));
    mu = mean(tmp_hyp(:));
    sig = std(tmp_hyp(:));
    tmp_hyp = (tmp_hyp-mu)/(4*sig)+0.5;
    tmp_hyp(tmp_hyp<0) = 0; tmp_hyp(tmp_hyp>1) = 1;
    tmp_fused = (tmp_fused-mu)/(4*sig)+0.5;
    tmp_fused(tmp_fused<0) = 0; tmp_fused(tmp_fused>1) = 1;
    hyp_rgb(:,:,b) = tmp_hyp;
    fused_rgb(:,:,b) = tmp_fused;
end
figure;
subplot(1,2,1); imshow(hyp_rgb); axis image; title('HS');
subplot(1,2,2); imshow(fused_rgb); axis image; title('Fused');
