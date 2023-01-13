# Purpose : Plot first two principal components of feature matrix

library(ggplot2);

# PC plot
pcPlot = function(D,colors){
  # D : Obs by predictors data matrix
  # colors : For classes

  if(missing(colors)){colors=c("royalblue","coral")};

  # Data
  X = data.matrix(D[,names(D) != "y"]);
  y = D$y;

  # PCs
  SVD = svd(x=X);
  PC = data.frame((X %*% (SVD$v))[,1:2]);
  colnames(PC) = c("Comp1","Comp2");

  PC$y = factor(y,levels=c(0,1),labels=c("CLD","HCC"));

  # Plot
  q = ggplot(data=PC,aes(x=Comp1,y=Comp2,color=y)) + geom_point(size=2);
  q = q + theme_bw() + labs(x="Component 1", y="Component 2");
  q = q + scale_color_manual(name="Disease",values=colors);

  Out = list("PCs"=PC,"Plot"=q);
  return(Out);
}

# Import
In = read.csv(file="~/Dropbox/6.867_Project/Data/Lasso_Subset.csv",stringsAsFactors=F);
# Format
ID = In[,"ID"];
In = In[,names(In) != "ID"];

Out = pcPlot(In)
show(Out$Plot);

g<-Out$Plot
g
pdf(file="~/Dropbox/6.867_Project/Final Report/filename.pdf",height=9,width=7)
print(g)
dev.off()
