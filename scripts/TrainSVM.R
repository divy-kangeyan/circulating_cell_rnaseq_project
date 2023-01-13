# Purpose : Train SVM model by repeated CV

library(caret);
library(doMC);
library(kernlab);
library(e1071);
library(doMC)

# Parallelize
k = detectCores();
registerDoMC(cores=k-2);

# Directory
setwd("~/Dropbox/6.867_Project/Data/");

# Import
In = read.csv(file="Lasso_Subset.csv",stringsAsFactors=F);
# Format
ID = In[,"ID"];
In = In[,names(In) != "ID"];
In$y = factor(In$y,levels=c(0,1),labels=c("CLD","HCC"));
X = In[,names(In) != "y"];
y = In[,"y"];

# Wrapper Function
TrainSVM = function(D,Test=F,Rep=20){
  # D : Data set
  # Test : Testing indicator
  # Rep : CV replicates
  
  # Split data
  y = D[,"y"];
  X = D[,names(D) != "y"];
  
  if(Test){
    # Train control
    TC = trainControl(method="repeatedcv", number=2, repeats=2, 
                 verboseIter=T, classProbs=T, 
                 selectionFunction="best");
    # Grid
    g = expand.grid("C"=2^c(from=-4,to=1),"sigma"=2^seq(from=-1,to=1))
  } else {
    # Train control
    TC = trainControl(method="repeatedcv", number=10, repeats=Rep, 
                      verboseIter=T, classProbs=T,
                      selectionFunction="best");
    # Grid
    g = expand.grid("C"=2^c(from=-8,to=1),"sigma"=2^seq(from=-5,to=5))
  }
  # Train
  svm.train = train(x=X, y=y, method="svmRadial",
              preProcess=c("center","scale"), 
              tuneGrid=g, trControl = TC, metric="Accuracy");
  ## Results
  
  # Tuning parameters
  tune = svm.train$bestTune;
  # Final model
  M = svm(x=X,y=y,scale=F,kernel="radial",type="C-classification",cost=tune$C,gamma=tune$sigma);
  # Final model performance
  A = svm.train$results;
  
  # Output
  Out = list("Model"=M,"Results"=A);
  return(Out);
}

# Train
Out = TrainSVM(In,R=4);
M = Out$Model;

# Prediction - categorical
yhat = predict(object=M,newdata=X,type="response");

# Prediction accuracy
mean(yhat == y);
