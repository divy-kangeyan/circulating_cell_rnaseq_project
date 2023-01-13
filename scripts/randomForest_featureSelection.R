setwd("~/Dropbox/6.867_Project/Data")

# Using random forest to perform feature selection
library(randomForest)
library(varSelRF)
library(Hmisc)
library(preprocessCore)

# Example from iris dataset
iris<-iris[sample(nrow(iris)),]
irisX<-iris[1:100,1:4]
irisY<-iris[1:100,5]
irisTestX<-iris[101:150,1:4]
irisTestY<-iris[101:150,5]
rfobj<-randomForest(irisX,irisY,keep.forest=TRUE)
pred<-predict(rfobj,irisTestX)





# Using out of bag error to select the variables
varSelObj<-varSelRF(irisX,irisY)

# Using the cancer data
load("trainData.RData")
trainData<-t(trainData)

Y<-rownames(trainData)
classes<-sub('[.]\\d+','',Y) # regex for the two classes
classes<-as.factor(classes)



# Preproces the training data
trainData <- trainData[(apply(trainData, 1, var)>0), ]
trainData <- log2(1+trainData)

# Quantile Normalization
names <- dimnames(trainData)
norm.target <- normalize.quantiles.determine.target(data.matrix(trainData),target.length=NULL)
trainData <- normalize.quantiles.use.target(data.matrix(trainData), target=norm.target, copy=FALSE)
dimnames(trainData) <- names

# Build random forest with the data for feature selection
rfobjCancer<-randomForest(trainData,importance=TRUE)

varImpMetric<-rfobjCancer$importance
giniImportance<-varImpMetric[,'MeanDecreaseGini']
ImpVarsInd<-giniImportance>0.001
RfFeatureSelectionData<-trainData[,ImpVarsInd]

save(RfFeatureSelectionData,file="/users/Divy/Dropbox/6.867_Project/Data/RfFeatureSelectionData.rda")

# Function to select the features based on the Gini threshold

selectRfData<-function(threshold){
  rfobjCancer<-randomForest(trainData,importance=TRUE)

  varImpMetric<-rfobjCancer$importance
  giniImportance<-varImpMetric[,'MeanDecreaseGini']
  ImpVarsInd<-giniImportance>threshold
  RfFeatureSelectionData<-trainData[,ImpVarsInd]
  save(RfFeatureSelectionData,file="RfFeatureSelectionData.RData")
  print('saved the rda file')

  }

# Used 0.005 as the cutoff to select the variables
selectRfData(0.005)

