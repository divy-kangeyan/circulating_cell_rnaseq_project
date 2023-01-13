library(caret)


N <- 20 # number of times to repeat cross-validation procedure
K <- 10 #number of folds

## data has samples in rows and features in columns
## outcomes is a vector (1 for HCC, 0 for CLD)
## params is a list of parameter values

## record number of correct predictions in array: 
correct <- array(NA, dim = c(K, length(params), N), dimnames = list(NULL, params , NULL))

for(j in 1:N){
  print(j)
  flds <- createFolds(as.factor(outcomes), k=K, list=T, returnTrain = F)
  for(i in 1:K){
    print(i)
    train <- data[-flds[[i]],]
    test <- data[flds[[i]],]
    for(p in 1:length(params)){
      ## Train Classifier
      
      ## Get number of correct preditions in test set
      correct[i, p ,j] <- ## store here
    }
  }
}