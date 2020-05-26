import os
from skimage import io
from skimage import data
from skimage.color import rgb2gray,rgba2rgb
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import glob
import pandas as pd
import operator
from skimage import img_as_ubyte


class figure:
	m=[]
	data=[]
	r1=0;r2=0;c1=0;c2=0;
	def __init__(self, mask,o):
		self.m=mask
		self.r1=min(mask[:,0])
		self.r2=max(mask[:,0])
		self.c1=min(mask[:,1])
		self.c2=max(mask[:,1])
		self.data = np.ones((28,28), dtype=np.float64)
		for bb in mask:
			self.data[bb[0]-self.r1+4,bb[1]-self.c1+4]=o[bb[0],bb[1]]
		self.pixel=255-img_as_ubyte(self.data)
		self.pixel=np.array(self.pixel).reshape(28*28)

image_list = []
cc=0
columns=[]
for x in range(28*28):
	columns.append('pixel'+ str(x))
df = pd.DataFrame(columns=columns)
for filename in glob.glob('capture/*.png'):
	print(filename)
	filename = os.path.join(filename)
	moon = io.imread(filename)
	moon= rgb2gray(rgba2rgb(moon))
	block_size = 11
	local_thresh = threshold_local(moon, block_size, offset=0.001)
	binary = moon >= local_thresh
	a=np.argwhere(binary==False)
	kmeans = KMeans(n_clusters=4, random_state=0).fit(a)
	b=[]
	for i in range(4):
		b.append(figure(a[kmeans.predict(a)==i],moon))
	b = sorted(b, key=operator.attrgetter('c1'))
	for i in range(4):
		im = Image.fromarray(img_as_ubyte(b[i].data))
		im.save("./figures/"+str(cc)+".jpg")
		df.loc[cc]=b[i].pixel
		cc=cc+1
df.to_csv("1.csv")
