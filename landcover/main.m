clear, close all

% addpath('sub_area_1');
addpath('CCF');
%% Load multimodal images
t1 = Tiff('EeteS_EnMAP_10m_sub_area1.tif','r'); % 300-360-242
Data_image = double(read(t1));
height = size(Data_image,1); width = size(Data_image,2);
Data_HSI = reshape(Data_image, height*width, []);

t2 = Tiff('Sentinel_2_sub_area1.tif','r');%300-360-12
Data_image = double(read(t2));
Data_MSI = reshape(Data_image, height*width, []);

t3 = Tiff('Sentinel_1_sub_area1.tif','r');%300-360-2
Data_image = double(read(t3));
Data_SAR = reshape(Data_image, height*width, []);

t4 = Tiff('3K_dsm_sub_area1.tif','r');%10000-12000
Data_image = double(read(t4));
Data_image = imresize(Data_image,[height, width]);
Data_DSM = reshape(Data_image, height*width, []);

clear t1 t2 t3 t4

Data = [Data_HSI, Data_MSI, Data_SAR, Data_DSM];
clear Data_HSI Data_MSI Data_SAR Data_DSM
%% Load groundtruth
t = Tiff('osm_landuse_sub_area1.tif','r');%1364-1636 double: 7219
Data_image = read(t);
Label_image = imresize(Data_image,[height, width],'nearest');

clear imageData t

%% Relabeling
Labels = unique(Label_image);
Counts = zeros(size(Labels));

j = 0; k = 0;
delLabel = 12;
for i = 1:size(Labels)
    map = (Label_image == Labels(i));
    Counts(i) = sum(map(:));
    if j == delLabel
        Label_image(map)=0;
        k = k - 1;
    else
        Label_image(map)=k;
    end
    j=j+1;
    k=k+1;
end

Label_1d = reshape(Label_image, height*width,1);

%% Split training & testing set
seed = 1;
rng(seed)
train_rate = 0.2;
train_idx_all = []; test_idx_all = [];
train_num = []; test_num = [];
maxLabel = max(Label_image(:));
for i = 1:maxLabel
    i_label = find(Label_1d==i);
    tmp_label = randperm(length(i_label));
    tmp = ceil(length(i_label)*train_rate);
    train_tmp = i_label(tmp_label(1:tmp));
    test_tmp = setdiff(i_label, train_tmp);
    train_idx_all = [train_idx_all;train_tmp];
    train_num = [train_num;length(train_tmp)];
    test_idx_all = [test_idx_all;test_tmp];
    test_num = [test_num;length(test_tmp)];
end

train_data = Data(train_idx_all,:);
train_label = Label_1d(train_idx_all,1);

test_data = Data(test_idx_all,:);
test_label = Label_1d(test_idx_all,1);

if test_idx_all 
    clear test_data
    test_data = Data;
end

%% Run CCF
nTrees = 100;
[M,oa2,pa2,~,kappa2,Predicts1D]=MyCCF2(train_data',train_label',test_data',test_label',nTrees, test_idx_all);
aa2=mean(pa2);
