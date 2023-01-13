library(caret)
library(randomForest)
library(Hmisc)
library(preprocessCore)
setwd("~/Dropbox/6.867_Project/Data")

N <- 20 # number of times to repeat cross-validation procedure
K <- 10 #number of folds

## data has samples in rows and features in columns
## outcomes is a vector (1 for HCC, 0 for CLD)
## params is a list of parameter values

## record number of correct predictions in array:
params <- c(seq(100,1000,100),1500)
correct <- array(NA, dim = c(K, length(params), N), dimnames = list(NULL, params , NULL))

load("RfFeatureSelectionData.RData")
data <- RfFeatureSelectionData
Y <- rownames(data)
classes <- sub('[.]\\d+','',Y) # regex for the two classes
outcomes <- as.factor(classes)


for(j in 1:N){
  print(j)
  flds <- createFolds(as.factor(outcomes), k = K, list = T, returnTrain = FALSE)
  for(i in 1:K){
    print(i)
    train <- data[-flds[[i]], ]

    Y <- rownames(train)
    classes <- sub('[.]\\d+', '', Y) # regex for the two classes
    trainClasses <- as.factor(classes)

    test <- data[flds[[i]], ]
    Y <- rownames(test)
    classes <- sub('[.]\\d+', '', Y) # regex for the two classes
    testClasses <- as.factor(classes)
    for(p in 1:length(params)){
      ## Train Classifier
      rfObj <- randomForest(train, trainClasses, ntree=params[p])
      #predictedClasses<-rfobjCancer$predicted
      #classificationError<-mean(classes==predictedClasses)
      validationPrediction <- predict(rfObj, test)
      #ClassificationErrorMatrix[i,j]<-mean(validationPrediction==trueClass)

      ## Get number of correct preditions in test set
      correct[i, p ,j] <- mean(validationPrediction == testClasses) ## store here
    }
  }
}

# Turn the code into a function
CVRandomForest <- function(data, outcomes, trees, varFrac, N, K){
  correct <- array(NA, dim = c(length(trees), length(varFrac), K, N), 
                   dimnames = list(trees, varFrac, NULL, NULL))
  for(j in 1:N){
    print(j)
    flds <- createFolds(as.factor(outcomes), k = K, list = T, returnTrain = F)
    for(i in 1:K){
      print(i)
      train <- data[-flds[[i]], ]

      Y <- rownames(train)
      classes <- sub('[.]\\d+', '', Y) # regex for the two classes
      trainClasses <- as.factor(classes)

      test <- data[flds[[i]],]
      Y <- rownames(test)
      classes <- sub('[.]\\d+', '', Y) # regex for the two classes
      testClasses <- as.factor(classes)
      for(p in 1:length(trees)){
          for (v in 1:length(varFrac)){
        ## Train Classifier
        rfObj <- randomForest(train, trainClasses, ntree=trees[p], mtry=varFrac[v])
        #predictedClasses<-rfobjCancer$predicted
        #classificationError<-mean(classes==predictedClasses)
        validationPrediction <- predict(rfObj, test)
        #ClassificationErrorMatrix[i,j]<-mean(validationPrediction==trueClass)

        ## Get number of correct preditions in test set
        correct[p, v, i, j] <- mean(validationPrediction == testClasses) ## store here
          }
      }
    }
  }
  return(correct)
}


# Training on the RF feature selection data
load("RfFeatureSelectionData.RData")
data <- RfFeatureSelectionData
Y <- rownames(data)
classes <- sub('[.]\\d+', '', Y) # regex for the two classes
outcomes <- as.factor(classes)
numVar = dim(data)[2]
roundednumVar <- round(sqrt(numVar), digits = -2)

numTrees <- c(seq(100, 1000, 100), 1500)
variableFraction <- c(roundednumVar * seq(0.1, 1, 0.1))

RFAccuracy <- CVRandomForest(data, outcomes, numTrees, 
                             variableFraction, 20, 10)


# Training on the univariate logistic regression feature selection data
load("LogRegFeatureSet.RData")
data <- LogRegFeatureSet
Y <- rownames(data)
classes <- sub('[.]\\d+','',Y) # regex for the two classes
outcomes <- as.factor(classes)
numVar = dim(data)[2]
roundednumVar <- round(sqrt(numVar),digits=-2)

numTrees <- c(seq(100,1000,100),1500)
variableFraction <- c(roundednumVar*seq(0.1,1,0.1))

LRAccuracy<-CVRandomForest(data,outcomes,numTrees,variableFraction,20,10)






# Use functions in caret to tune the parameter
# Test the optimal parameters in the test set


# Custome random fores to tune both ntrees and mtry


customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes





# train model
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Training on the LASSO feature selection data
ptm<-proc.time()
data<-read.csv('Lasso_Subset.csv',check.names = FALSE)
data<-data[,!(names(data)%in%"ID")]

X<-data[,!(names(data)%in%"y")]
LASSOFeatures<-colnames(X)

data<-load('trainDataNorm.RData')
data<-trainDataNorm
colnames(data)<-sub('-','.',colnames(data))
X<-data[,LASSOFeatures]
Y<-rownames(X)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)


numVar = dim(X)[2]
roundednumVar<-2*round(sqrt(numVar),digits=2)

numTrees<-c(seq(100,1000,100))
numVariable<-round(c(roundednumVar*seq(0.1,1,0.1)))

control <- trainControl(method="repeatedcv", number=10, repeats=20)
tunegrid <- expand.grid(.mtry=numVariable, .ntree=numTrees)
customLASSO <- train(x=X,y=Y, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
print(proc.time() - ptm)
summary(customLASSO)


RFTuningMatrix<-(customLASSO$results)[,c(1,2,3)]
write.table(RFTuningMatrix,file='RFTuningDf.txt')


# Training the RF feature selection data
load("RfFeatureSelectionData.RData")
data<-RfFeatureSelectionData
Y<-rownames(data)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)

X<-data

numVar = dim(data)[2]
roundednumVar<-2*round(sqrt(numVar),digits=2)

numTrees<-c(seq(100,1000,100))
numVariable<-round(c(roundednumVar*seq(0.1,1,0.1)))


control <- trainControl(method="repeatedcv", number=10, repeats=20)
tunegrid <- expand.grid(.mtry=numVariable, .ntree=numTrees)
customRFFeatures <- train(x=X,y=Y,data=data, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
summary(customRFFeatures)


# Training the Univariate-LR feature selection data
load("LogRegFeatureSet.RData")
data<-LRFeatureSelectionData
Y<-rownames(data)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)

X<-data

numVar = dim(data)[2]
roundednumVar<-2*round(sqrt(numVar),digits=2)

numTrees<-c(seq(100,1000,100))
numVariable<-round(c(roundednumVar*seq(0.1,1,0.1)))


control <- trainControl(method="repeatedcv", number=10, repeats=20)
tunegrid <- expand.grid(.mtry=numVariable, .ntree=numTrees)
customLRFeatures <- train(x=X,y=Y,data=data, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
summary(customLRFeatures)
stopCluster(cl)

# Get the data from the cluster
load('RFObj.RData')

###
LRtrainResults<-customLRFeatures$results
RFtrainResults<-customRFFeatures$results

head(LASSOtrainResults,n=23)

# Load the testing Data
load('testData.RData')

# Preprocess the test data
names <- dimnames(testData)
norm.target <- normalize.quantiles.determine.target(data.matrix(testData),target.length=NULL)
testData <- normalize.quantiles.use.target(data.matrix(testData), target=norm.target, copy=FALSE)
dimnames(testData) <- names
testData<-t(testData)
testData<-as.data.frame(testData)
Y<-rownames(testData)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
testDataClasses<-as.factor(Y)


# Train the model with the best tuning parameter
# For LASSO the best parameters chosen were ntree = 1000, mtry = 1
data<-load('trainDataNorm.RData')
data<-trainDataNorm
colnames(data)<-sub('-','.',colnames(data))
X<-data[,LASSOFeatures]
Y<-rownames(X)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)


trLASSOObj<-randomForest(x=X,y=Y,ntree=500,mtry=1)
trainPrediction_LASSO<-trLASSOObj$predicted
LASSOTrainAccuracy<-mean(trainPrediction_LASSO==Y)

# Test the model on the test set
testData<-load('testDataNorm.RData')
testData<-testDataNorm
colnames(testData)<-sub('-','.',colnames(testData))
testData<-testData
testData<-testData[,LASSOFeatures]

testPrediction_LASSO<-predict(trLASSOObj,testData)
LASSOTestAccuracy<-mean(testPrediction_LASSO==testDataClasses)




# For RF the best parameters chosen were ntree = 200, mtry = 79
load("RfFeatureSelectionData.RData")
data<-RfFeatureSelectionData
Y<-rownames(data)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)

X<-data
RFFeatures<-colnames(X)

trRFObj<-randomForest(x=X,y=Y,ntree=200,mtry=79)
trainPrediction_RF<-trRFObj$predicted
RFTrainAccuracy<-mean(trainPrediction_RF==Y)


# Test the model on the test set
testData<-load('testDataNorm.RData')
testData<-testDataNorm
testData<-testData
testData<-testData[,RFFeatures]

testPrediction_RF<-predict(trRFObj,testData)
RFTestAccuracy<-mean(testPrediction_RF==testDataClasses)




# For LR the best parameters chosen were ntree = 1000, mtry = 130
load("LogRegFeatureSet.RData")
data<-LogRegFeatureSet
Y<-rownames(data)
Y<-sub('[.]\\d+','',Y) # regex for the two classes
Y<-as.factor(Y)

X<-data
LRFeatures<-colnames(X)
trLRObj<-randomForest(x=X,y=Y,ntree=1000,mtry=130)
trainPrediction_LR<-trLRObj$predicted
LRTrainAccuracy<-mean(trainPrediction_LR==Y)


testData<-load('testDataNorm.RData')
testData<-testDataNorm
testData<-testData
testData <- testData[,LRFeatures]

testPrediction_LR <- predict(trLRObj,testData)
LRTestAccuracy <- mean(testPrediction_LR==testDataClasses)




