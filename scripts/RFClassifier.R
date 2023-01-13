# Random forest classification
# Two potential tuning parameters in RF are:
# Number of trees (ntree) built and fraction of variable to choose when building a tree (mtry) 


Trees = c(seq(100,1000,100),1500)
numVar = dim()
roundednumVar<-round(sqrt(numVar),digits=-2)
varFrac = c(roundednumVar*seq(0.1,1,0.1))

ClassificationErrorMatrix<-matrix(nrow=length(Trees),ncol=length(numVar))

for (i in 1:length(Trees)){
  for (j in 1:length(varFrac)){
    rfObj<-randomForest(X,classes,ntree=Trees[i],varFrac[j])
    #predictedClasses<-rfobjCancer$predicted
    #classificationError<-mean(classes==predictedClasses)
    validationPrediction<-prediction(rfObj,LeftOutData)
    ClassificationErrorMatrix[i,j]<-mean(validationPrediction==trueClass)
    
  }
  
}

