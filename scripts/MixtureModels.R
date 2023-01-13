library(mixtools)
library(mclust)

# Using mclust package for GMM
data(iris)
class = iris$Species
table(class)

set.seed(9900)
randSample<-sample(nrow(iris))
iris<-iris[randSample,]  # randomize the sample

X = iris[,1:4]
trueLabel<-iris$Species

TrainX<-X[1:(nrow(X)*0.7),]
TrainLabel<-trueLabel[1:(nrow(X)*0.7)]

TestX<-X[((nrow(X)*0.7)+1):150,]
TestLabel<-trueLabel[((nrow(X)*0.7)+1):150]

mod2 = MclustDA(TrainX, TrainLabel, modelType = "EDDA") 

# Modeltype here EDDA means single component for each class with the same
# covariance sructure among classes
summary(mod2)

plot(mod2, what = "scatterplot")

# Corss validation error
unlist(cvMclustDA(mod2, nfold = 10)[2:3])

predictions<-predict(mod2)

# Performance in the testing data
testPredictions<-predict(mod2,TestX)


# Cancer Data
load('/Users/Divy/Dropbox/6.867_Project/Data/trainData.RData')
load('/Users/Divy/Dropbox/6.867_Project/Data/testData.RData')
trainData<-t(trainData)

Y<-rownames(trainData)
classes<-sub('[.]\\d+','',Y) # regex for the two classes
classes<-as.factor(classes)

trainData<-trainData[,1:10]

mod2 = MclustDA(trainData, classes, modelType = "EDDA")
