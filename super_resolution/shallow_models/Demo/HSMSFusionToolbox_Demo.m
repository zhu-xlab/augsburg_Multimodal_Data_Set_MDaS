clear all
close all
clc

% A demo program for
% N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
% multispectral data fusion: a comparative review of the recent 
% literature," IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 2, 
% pp. 29-56, June 2017.
            
%% Load data

scaling = 10000;
toolbox_path = 'C:\Users\Jingliang\Documents\projects\Augsburg\super_resolution\HSMSFusionToolbox\'; % path where you put HSMSFusionToolbox
data_path = [toolbox_path 'HSMSFusionToolbox\Data\HyperspecVNIRChikusei\'];

PRECISION    = 'int16';
INTERLEAVE   = 'bsq';
BYTEORDER    = 'ieee-le';
OFFSET       =  0;

% Multispectral data
rows1 = 540;
cols1 = 420;
bands1 = 8;
FILENAME_MS = 'MSI/Headwall_Chikusei_540x420_WV2_MSHR_SNR35.dat'; 
SIZE_MS     = [rows1,cols1,bands1];
MSI        = double(multibandread([data_path FILENAME_MS], SIZE_MS, PRECISION, OFFSET, INTERLEAVE, BYTEORDER))/scaling;

% Hyperspectral data
rows2 = 90;
cols2 = 70;
bands2 = 128;
FILENAME_HS = 'HSI/Headwall_Chikusei_540x420_HSLR_fDS6_SNR35.dat';
SIZE_HS     = [rows2,cols2,bands2];
HSI        = double(multibandread([data_path FILENAME_HS], SIZE_HS, PRECISION, OFFSET, INTERLEAVE, BYTEORDER))/scaling;

% Reference
FILENAME_REF = 'REF/Headwall_Chikusei_540x420_HSHR_ref_denoised.dat'; 
SIZE_REF     = [rows1,cols1,bands2];
REF        = multibandread([data_path FILENAME_REF], SIZE_REF, PRECISION, OFFSET, INTERLEAVE, BYTEORDER);
REF = double(REF)/scaling;

ratio = rows1/rows2; 
    
HSI(HSI<0) = 0;
MSI(MSI<0) = 0;

%% Fusion
% Methods
% 'GSA'    : Gram-Schmidt Adaptive
% 'SFIMHS' : Smoothing filter-based intensity modulation with hypersharpening
% 'GLPHS'  : Generalized Laplacian pyramid with hypersharpening
% 'CNMF'   : Coupled nonnegative matrix factorization
% 'ECCV14' : Akhtar's method presented in ECCV'14
% 'ICCV15' : Lanaras's method presented in ICCV'15
% 'HySure' : Hyperspectral superresolution
% 'MAPSMM' : Eismann's method
% 'FUSE'   : Fast fusion based on Sylvester equation

method = 'GSA';

disp(['Process ' method ' ...']);

tic
switch method
    case 'GSA'
        Out = GSA_wrapper(HSI,MSI,ratio);
    case 'SFIMHS'
        m = 2;
        Out = SFIM(HSI,MSI,m);
    case 'GLPHS'
        m = 2;
        Out = MTF_GLP_wrapper(HSI,MSI,ratio,m);
    case 'CNMF'
        Out = CNMF_fusion(HSI,MSI);
    case 'ECCV14'
        Out = Akhtar_wrapper(HSI,MSI);
    case 'ICCV15'
        Out = Lanaras_wrapper(HSI,MSI,REF);
    case 'HySure'
        overlap = repmat(1:bands2,[bands1 1]); 
        Out = HySure_wrapper(HSI, MSI, ratio, overlap, 30, round(ratio/2)-1);
    case 'MAPSMM'
        Out = MAPSMM_fusion(HSI,MSI);
    case 'FUSE'
        Out = FUSE_wrapper(HSI,MSI,ratio,scaling);
end
toc

%% Denoising
switch method
    case {'GSA','SFIMHS','GLPHS','MAPSMM'}
        disp('Perform denoising ...')
        [Out] = denoising(Out);
    case {'CNMF','ECCV14','ICCV15','HySure','FUSE'}
        disp('No denoising');
end

%% Save fusion results
multibandwrite(Out*scaling,[data_path 'FusionResults/' method],INTERLEAVE);

%% Quantitative evaluation
quality = QualityIndices(Out,REF,ratio);