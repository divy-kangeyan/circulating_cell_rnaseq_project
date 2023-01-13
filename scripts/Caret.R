library(caret);
library(foreach);
library(ggplot2);
library(mvnfast);

########################
# Simulate data
########################

# Observations
n = 100;
# Parameters
mu1 = c(0,0);
mu2 = c(2,2);
sigma1 = matrix(c(1,0.5,0.5,1),nrow=2,byrow=T);
sigma2 = matrix(c(1,-0.2,-0.2,1),nrow=2,byrow=T);
# Membership probability
p1 = 0.35;
p2 = 1 - p1;
# Simulate
D = foreach(i=1:n,.combine=rbind) %do% {
  # Cluster
  z = 1 + rbinom(n=1,size=1,prob=p2);
  # Covariate
  if(z==1){
    x = rmvn(n=1,mu=mu1,sigma=sigma1);
  } else {
    x = rmvn(n=1,mu=mu2,sigma=sigma2);
  }
  # Out
  Out = data.frame(t(c(z,x)));
  colnames(Out) = c("y","x1","x2");
  return(Out);
}

# Plot
q = ggplot(D,aes(x=x1,y=x2,color=factor(y))) + geom_point();
q = q + theme_bw() + scale_color_manual(name="Class",values=c("steelblue","coral"));
show(q);

########################
# Data Splitting
########################

# Get complement of testing set
setComp = function(a,n){
  # a : Set
  # n : Obs
  
  omega = seq(1:n);
  b = omega[!(omega %in% a)];
  return(b);
}

n = nrow(D);
Splits = createFolds(seq(1:n),k=5);

a = Splits$Fold1;
b = setComp(a,n);

Dtrain = D[b,];
Dtest = D[a,];

########################
# SVM
########################

## SVM Fitting
# Training control
# Note : Can specify custom summary functions 
TC = trainControl(method="cv", repeats=10, selectionFunction="best");
# TC = trainControl(method="cv", repeats=10, selectionFunction="oneSE");
set.seed(100); 
svm = train(form=factor(y) ~ ., data=Dtrain, method="svmRadial",
            preProcess=c("center","scale"), 
            tuneLength=10, trControl = TC);
# Plot effect of tuning parameter
ggplot(svm, metric="Accuracy") + theme_bw();
# True labels
ytrue = Dtest$y;
# Predictions
p1 = predict(svm, newdata=Dtest);

########################
# Logistic Regression
########################

set.seed(100);
glm1 = train(form=factor(y) ~ ., data=Dtrain, method="glm", metric="Accuracy",
             preProcess=c("center","scale"), trControl = TC);
# Predictions
p2 = predict(glm1, newdata=Dtest);
# p2 = predict(glm1, newdata=Dtest, type="prob");

library("pROC");
# Optimize ROC rather than accuracy
# Note : train prefers factors to have character names
Dtrain$y = factor(Dtrain$y,levels=c(1,2),labels=c("a","b"));
TC2 = trainControl(method="cv", repeats=10, 
                  summaryFunction=twoClassSummary, classProbs=T);
set.seed(100); 
glm2 = train(form=y ~ ., data=Dtrain, method="glm",
            preProcess=c("center","scale"), 
            tuneLength=10, trControl = TC2, metric="ROC");

## Model comparison
# Note: Requires same cv folds

# Summary of performance on cv folds
comp = resamples(x=list("svm"=svm,"log"=glm1));
summary(comp);

# Test for performance differences
delta = diff(comp);
summary(delta);

########################
# Neural Networks
########################

## Tuning options
# Decay : Regularization parameter
# Size : Hidden units
grid = expand.grid(.decay = 10^seq(from=-1,to=-5),.size=c(5,10,100));
TC = trainControl(method="cv", repeats=5, selectionFunction="best");

nnet1 = train(form=factor(y)~., data=Dtrain, method="nnet",
              preProc = c("center","scale"), trace = F, maxit = 1000,
              tuneGrid = grid, trControl = TC);

# Predictions
p1 = predict(nnet1, newdata=Dtest);
