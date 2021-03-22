clear all
close all
%clc
addpath(genpath('.'))
% Code from:
% N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
% multispectral data fusion: a comparative review of the recent 
% literature," IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 2, 
% pp. 29-56, June 2017.
            
%% Load data
region = 'sub_area_1';
toolbox_path = '.\'; 
% data path
data_path = [toolbox_path '..\data\' region '\'];
switch region
    case 'sub_area_1'        
        % Hyperspectral data
        FILENAME_HS  = 'EeteS_EnMAP_30m_sub_area1.tif';
        % Multispectral data
        FILENAME_MS  = 'EeteS_Sentinel_2_10m_sub_area1.tif'; 
        % Reference
        FILENAME_REF = 'EeteS_EnMAP_10m_sub_area1.tif'; 
   case 'sub_area_2'
   case 'sub_area_3'   
        
end

% Hyperspectral data
HSI = readgeoraster([data_path FILENAME_HS]);
bands_HSI = 1:210;
HSI = double(HSI(:,:,bands_HSI));

% Multispectral data
MSI = readgeoraster([data_path FILENAME_MS]);
% bands_MSI = [2,3,4,8]; MSI(:,:,bands_MSI)
MSI = double(MSI);

% Reference
REF = readgeoraster([data_path FILENAME_REF]);
REF = double(REF(:,:,bands_HSI));


%% data preprocessing
% reflectance correction
% MSI(MSI(:)>1e4) = 0;
% HSI(HSI(:)>1e4) = 0;
% REF(REF(:)>1e4) = 0;

% HSI = HSI_sub; clear HSI_sub
% MSI = MSI_sub; clear MSI_sub
% REF = REF_sub; clear REF_sub

% normalization/scaling
% HSI_scaling = max(HSI(:));
% MSI_scaling = max(MSI(:));
% REF_scaling = max(REF(:));
% HSI = HSI/HSI_scaling;
% MSI = MSI/MSI_scaling;
% REF = REF/REF_scaling;

scaling = 10000;
HSI = HSI/scaling;
MSI = MSI/scaling;
REF = REF/scaling;


[rows2, cols2, bands2] = size(HSI);
[rows1, cols1, bands1] = size(MSI);
[rows1, cols1, bands2] = size(REF);
ratio = rows1/rows2;

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

method = 'CNMF';

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
[~,r] = readgeoraster([data_path FILENAME_REF]);
info  = geotiffinfo([data_path FILENAME_REF]);
key = info.GeoTIFFTags.GeoKeyDirectoryTag;
geotiffwrite([data_path 'FusionResults\' region '_' method '.tif'], Out*scaling, r,'GeoKeyDirectoryTag',key);


%% Quantitative evaluation
quality = QualityIndices(Out,REF,ratio);