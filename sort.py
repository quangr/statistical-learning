from scipy import ndimage
import os
from skimage import io
from skimage import transform
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_local
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import glob
import pandas as pd
import operator
from skimage import img_as_ubyte
import graph

class figure:
    m = []
    data = []
    r1 = 0
    r2 = 0
    c1 = 0
    c2 = 0

    def __init__(self, mask, o,o1):
        self.m = mask
        self.r1 = min(mask[:, 0])
        self.r2 = max(mask[:, 0])
        self.c1 = min(mask[:, 1])
        self.c2 = max(mask[:, 1])
        t = np.zeros((26, 26,4),dtype=np.uint8)
        self.data = np.ones((26, 26), dtype=np.float64)
        for bb in mask:
            self.data[bb[0] - self.r1 + 4, bb[1] - self.c1 + 4] = o[bb[0], bb[1]]
            t[bb[0] - self.r1 + 4, bb[1] - self.c1 + 4] = o1[bb[0], bb[1]]
        self.pixel = 255 - img_as_ubyte(self.data)
        lc = np.sum((self.pixel > np.max(self.pixel) / 2).T * np.arange(26)) / np.sum(
            self.pixel > np.max(self.pixel) / 2) - 13
        wc = np.sum((self.pixel > np.max(self.pixel) / 2) * np.arange(26)) / np.sum(
            self.pixel > np.max(self.pixel) / 2) - 13
        tt= transform.SimilarityTransform(translation=(-int(np.round(lc)), -int(np.round(wc))))
        self.data1 = transform.warp(t, tt)


a = pd.read_csv("./mix.csv")
y = a[a.columns[0]]

a = a.drop(a.columns[0], 1)
imgs = a.to_numpy()
imgs = np.array(imgs > 0, dtype=np.double)
X=imgs
theta = np.zeros(imgs.shape[0], dtype=np.double)
for kk in range(104,X.shape[0]):
    Nx=sum(y==y[kk])
    nX=np.zeros([Nx+2,X.shape[1]])
    nX[:Nx,] = X[y==y[kk]]
    nX[Nx+0,:]=ndimage.rotate(X[kk].reshape([26, 26]), -4, reshape=False).reshape(26*26)
    nX[Nx+1,:]=ndimage.rotate(X[kk].reshape([26, 26]), 4, reshape=False).reshape(26*26)
    co=graph.isocor(nX)
    lmin = np.min(co, 0)[0]
    lmax = np.max(co, 0)[0]
    hh=(lmin+lmax)/2
    if np.abs(co[-2, 0] - hh) > np.abs(co[-1, 0] - hh):
        nX = np.zeros([Nx + 12, X.shape[1]])
        nX[:Nx, ] = X[y == y[kk]]
        for g in range(12):
            nX[Nx + g, :] = ndimage.rotate(X[kk].reshape([26, 26]), g*5+5, reshape=False).reshape(26 * 26)
        co = graph.isocor(nX)
        co=co[-12:]
        theta[kk]=5*np.argmin(np.abs(co[:,0] - hh))+1*5
    else:
        nX = np.zeros([Nx + 12, X.shape[1]])
        nX[:Nx, ] = X[y == y[kk]]
        for g in range(12):
            nX[Nx + g, :] = ndimage.rotate(X[kk].reshape([26, 26]), -g*5-5, reshape=False).reshape(26 * 26)
        co = graph.isocor(nX)
        co=co[-12:]
        theta[kk]=-5*np.argmin(np.abs(co[:,0] - hh))-1*5
cc=0
for filename in glob.glob('./test/*.png'):
    print(filename)
    filename = os.path.join(filename)
    moon = io.imread(filename)
    moon1=moon
    bn=np.zeros(moon1.shape,np.double)
    moon = rgb2gray(rgba2rgb(moon))
    block_size = 11
    local_thresh = threshold_local(moon, block_size, offset=0.001)
    binary = moon >= local_thresh
    a = np.argwhere(binary == False)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(a)
    b = []
    for i in range(4):
        b.append(figure(a[kmeans.predict(a) == i], moon,moon1))
    b = sorted(b, key=operator.attrgetter('c1'))
    for i in range(4):
        bn[12:38,10+i*30:10+i*30+26]=transform.rotate(b[i].data1,theta[cc*4+i+104])
        im = Image.fromarray((transform.rotate(b[i].data1,theta[cc*4+i+104]) * 255).astype(np.uint8))
        im.save("./sort/fig/" + str(cc*4+i+104) + ".png")
    im = Image.fromarray((bn*255).astype(np.uint8))
    im.save("./sort/com/" + str(cc) + ".png")
    cc = cc + 1
