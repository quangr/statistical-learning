library(jpeg)
library(grid)
library(readr)
library(ggplot2)
setwd("~/statistical learning/statistical-learning/figures")
dr <- read_csv("~/statistical learning/statistical-learning/LLE.csv")
dr$x=as.factor(dr$x)
ggplot(dr[dr$x==9|dr$x==0,], aes(x=`0`, y=`1`,color=x)) + geom_point()
idr=dr
idr[2:3]=scale(dr[2:3])
a<-ggplot(idr, aes(x=`0`, y=`1`))+  geom_point()
for (i in 0:100) {
  if (1|idr$x[i+1]==2){
    a<- a + annotation_raster(readJPEG(paste0("./",toString(i),".jpg")), xmin = idr$`0`[i+1], ymin = idr$`1`[i+1],xmax = 0.08+idr$`0`[i+1], ymax =  0.08+idr$`1`[i+1])
  }
}

