library(jpeg)
library(grid)
library(readr)
library(ggplot2)
setwd("~/statistical learning/statistical-learning/sort")
dr <- read_csv("~/statistical learning/statistical-learning/isomap.csv")
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

setwd("~/statistical learning/statistical-learning/sort")
or <- read_csv("~/statistical learning/statistical-learning/mix.csv")
dr <- read_csv("~/statistical learning/statistical-learning/isomap.csv")
idr=dr
idr[2:11]=scale(dr[2:11])
a<-ggplot(idr, aes(x=`8`, y=`9`))+  geom_point()
for (i in 0:(nrow(dr)-1)) {
    a<- a + annotation_raster(readJPEG(paste0("./",toString(which(or$y==7)[i+1]-1),".jpg")), xmin = idr$`8`[i+1], ymin = idr$`9`[i+1],xmax = 0.08+idr$`8`[i+1], ymax =  0.08+idr$`9`[i+1])
}


setwd("~/statistical learning/statistical-learning/sort")
or <- read_csv("~/statistical learning/statistical-learning/mix.csv")
dr <- read_csv("~/statistical learning/statistical-learning/LLE.csv")
idr=dr
idr[2:11]=scale(dr[2:11])
a<-ggplot(idr, aes(x=`0`, y=`1`))+  geom_point()
for (i in 0:(nrow(dr)-1)) {
  a<- a + annotation_raster(readJPEG(paste0("./",toString(which(or$y==7)[i+1]-1),".jpg")), xmin = idr$`0`[i+1], ymin = idr$`1`[i+1],xmax = 0.08+idr$`0`[i+1], ymax =  0.08+idr$`1`[i+1])
}

setwd("~/statistical learning/statistical-learning/sort")
dr <- read_csv("~/statistical learning/statistical-learning/isomap.csv")
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

