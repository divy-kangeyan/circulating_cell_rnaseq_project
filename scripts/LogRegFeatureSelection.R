setwd("~/Dropbox/6.867_Project/Data")
load("trainData.RData")
load("testData.RData")

library(preprocessCore)
library(Hmisc)

data <- trainData
test <- testData

## Only keep features w/ nonzero variance
data <- data[(apply(data, 1, var)>0), ]
test <- test[rownames(data),]

## log 2 transformation
data <- log2(1+data)
test <- log2(1+test)

## Quantile Normalization
names <- dimnames(data)
names.test <- dimnames(test)
norm.target <- normalize.quantiles.determine.target(data.matrix(data),target.length=NULL)
data <- normalize.quantiles.use.target(data.matrix(data), target=norm.target, copy=F)
test <- normalize.quantiles.use.target(data.matrix(test), target=norm.target, copy=F)
dimnames(data) <- names

## Make outcomes
y <- as.numeric(substring(colnames(data),1,3)=="HCC")

## Univariate logistic regression feature selection
x <- t(data)
features <- numeric(0)
for(i in 1:ncol(x)){
  max_by_y = summarize(x[,i], by=y, FUN=max)
  min_by_y = summarize(x[,i], by=y, FUN=min)
  if(max_by_y[1,2] <= min_by_y[2,2] || max_by_y[2,2] <= min_by_y[1,2]){
    features <- c(features, i)
  } else{
    fit <- glm(y~x[,i], family=binomial())
    if(summary(fit)$coefficients[2,4] <= 0.05){
      features <- c(features, i)
    }
  }
}


LogRegFeatureSet <- x[,features]
save(LogRegFeatureSet, file="LogRegFeatureSet.RData")
LogRegTestSet <- t(test)[,colnames(LogRegFeatureSet)]
save(LogRegTestSet, file="LogRegTestSet.RData")


trainDataNorm <- t(data)
testDataNorm <- t(test)
save(trainDataNorm, file="trainDataNorm.RData")
save(testDataNorm, file="testDataNorm.RData")
