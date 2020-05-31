library(jpeg)
library(grid)
isomap <- read_csv("~/statistical learning/statistical-learning/isomap.csv")
setwd("~/statistical learning/statistical-learning/figures")
isomap$x=as.factor(isomap$x)
ggplot(isomap[isomap$x==9|isomap$x==0,], aes(x=`0`, y=`1`,color=x)) + geom_point()
iisomap=isomap
iisomap[2:3]=scale(isomap[2:3])
a<-ggplot(iisomap, aes(x=`0`, y=`1`))+  geom_point()
for (i in 0:100) {
	if (iisomap$x[i+1]==2){
  a<- a + annotation_raster(readJPEG(paste0("./",toString(i),".jpg")), xmin = iisomap$`0`[i+1], ymin = iisomap$`1`[i+1],xmax = 0.08+iisomap$`0`[i+1], ymax =  0.08+iisomap$`1`[i+1])
}
}

