function [ M,oa,pa,ua,kappa,CCF]=ThreeClassifers(traindata,TrainLabel,testdata,TestLabel,flag)
% read data
% load('TR.mat')
% load('TRL.mat')
% 
% load('TE.mat')
% load('TEL.mat')


test = testdata';
teLab = TestLabel';

train = traindata';
trLab = TrainLabel';

switch flag
    case 0
%% linear svm
% W=[train;test];
% [ W,m,s ] = featureNormalise( W',0 );% feature scaling for training sampels
% W=W';
% trainNormalised=W(1:size(train,1),:);
% testNormalised=W(1+size(train,1):end,:);
% W=[train;test];
% [ trainNormalised,m,s ] = featureNormalise( train,3 );% feature scaling for training sampels
% [ trainNormalised,m,s ]=zscore(train);
% testNormalised = (test - repmat(m,size(test,1),1))./repmat(s,size(test,1),1);% feature scaling for test samples
% trainNormalised=W(1:size(train,1),:);
% testNormalised=W(1+size(train,1):end,:);
trainNormalised=train;
testNormalised=test;

% train SVM model
folds = 5;
[ cv_C, ~ ] = linearSVMPar( trLab,trainNormalised,folds );% optimal parameters searching
model = svmtrain2(trLab,trainNormalised,sprintf('-c %f -t %f', cv_C, 0));% SVM training
 

%
% Step 3: evaluation on test sample
%
[classTest] = svmpredict2(teLab, testNormalised, model);% predict using trained model



[ M,oa,pa,ua,kappa ] = confusionMatrix( teLab, classTest );

    case 1
%% gaussion kernel svm

% feature normalization
W=[train;test];
 for i=1:size(W,2)
     W(i,:)=double(mat2gray(W(i,:)));
 end
% [ trainNormalised,m,s ] = featureNormalise( train,3 );% feature scaling for training sampels
% [ trainNormalised,m,s ]=zscore(train);
% testNormalised = (test - repmat(m,size(test,1),1))./repmat(s,size(test,1),1);% feature scaling for test samples
trainNormalised=W(1:size(train,1),:);
testNormalised=W(1+size(train,1):end,:);
% m=mean(train,1);
% trainNormalised=train-repmat(m,size(train,1),1);
% testNormalised = test - repmat(m,size(test,1),1);
% trainNormalised=train;
% testNormalised=test;
% train SVM model
folds = 5;
[ cv_C, cv_G, ~ ] = kenelPar( trLab,trainNormalised,folds );% optimal parameters searching
model = svmtrain2(trLab,trainNormalised,sprintf('-c %f -g %f', cv_C, cv_G));% SVM training

%
% Step 3: evaluation on test sample
%
[classTest] = svmpredict2(teLab, testNormalised, model);% predict using trained model

%
% evaluation on test sample
%

[ M,oa,pa,ua,kappa ] = confusionMatrix( teLab, classTest );


case 2
% CCF
% W=[train;test];
%  for i=1:size(W,2)
%      W(i,:)=double(mat2gray(W(i,:)));
%  end
% % % % [ W,m,s ] = featureNormalise( W',0 );% feature scaling for training sampels
% % % % W=W';
% trainNormalised=W(1:size(train,1),:);
% testNormalised=W(1+size(train,1):end,:);

% feature normalization
% [ trainNormalised,m,s ] = featureNormalise( train,3);% feature scaling for training sampels
% [ trainNormalised,m,s ]=zscore(train);
% testNormalised = (test - repmat(m,size(test,1),1))./repmat(s,size(test,1),1);% feature scaling for test samples
% X=[train;test];
% X=Max_Min_Normlization(X);
% trainNormalised=X(1:size(train,1),:);
% testNormalised=X(size(train,1)+1:end,:);
trainNormalised=train;
testNormalised=test;
% m=mean(train,1);
% trainNormalised=train-repmat(m,size(train,1),1);
% testNormalised = test - repmat(m,size(test,1),1);
% [ trainNormalised,m,s ] = featureNormalise( train,0 );% feature scaling for training sampels
% % trainNormalised=trainNormalised';
% [ trainNormalised,m,s ]=zscore(train);
% testNormalised = (test - repmat(m,size(test,1),1))./repmat(s,size(test,1),1);% feature scaling for test samples
% trainNormalised=W(1:size(train,1),:);
% testNormalised=W(1+size(train,1):end,:);


% W=[train;test];
%  for i=1:size(W,2)
%      W(i,:)=double(mat2gray(W(i,:)));
%  end
% % [ W,m,s ] = featureNormalise( W',0 );% feature scaling for training sampels
% % W=W';
% trainNormalised=W(1:size(train,1),:);
% testNormalised=W(1+size(train,1):end,:);

nTrees = 300;

% train ccf model
[CCF] = genCCF(nTrees,trainNormalised,trLab);


%
% Step 3: evaluation on test sample
%
[classTest] = predictFromCCF(CCF,testNormalised);% predict using trained model



[ M,oa,pa,ua,kappa ] = confusionMatrix( teLab, classTest );
end
end






