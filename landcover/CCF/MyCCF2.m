function [M,oa,pa,ua,kappa,forestPredicts]=MyCCF2(traindata,TrainLabel,testdata,TestLabel,nTrees,test_idx)%only for test

train = traindata';
trLab = TrainLabel';
test = testdata';
teLab = TestLabel';
CCF = genCCF(nTrees,train,trLab); %train ccf model
[forestPredicts, forestProbs, treePredictions, cumulativeForestPredicts] = predictFromCCF(CCF,test); %predict using trained model
[M,oa,pa,ua,kappa] = confusionMatrix( teLab, forestPredicts(test_idx)); %%evaluation on test sample

end