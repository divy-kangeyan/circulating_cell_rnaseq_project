# Directory
flag = file.exists("testData.RData");
if(!flag){setwd("~/Dropbox/6.867_Project/Data/")};
library(glmnet);

# Import
load("trainData.RData");
Data = trainData;

# Variance filter
ind = (apply(Data,MARGIN=1,FUN=var)>0);
Data = Data[ind,];

# Outcomes
z = colnames(Data);
y = rep(0,length(z));
y[grep(pattern="^HCC.*",x=z)]=1;
# Check : z[as.logical(y)]

# LASSO
X = data.matrix(t(Data));
M1 = glmnet(y=y,x=X,family="binomial",alpha=0.95,nlambda=101);

# Selected predictors
lambda = M1$lambda[which.max(M1$df)];
Cols = predict(object=M1,s=lambda,type="nonzero")$X1;
Sub = data.frame(X[,Cols]);

# Output
Sub$y = y;
Sub$ID = row.names(Sub);
Sub = Sub[order(Sub$ID),];
write.csv(Sub,file="Lasso_Subset.csv",row.names=F);
