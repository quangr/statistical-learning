import os
from skimage import io
from skimage import data
from skimage.color import rgb2gray,rgba2rgb
from skimage.filters import threshold_local
import numpy as np
from PIL import Image
import pandas as pd
import operator
from skimage import img_as_ubyte



def TwoDPCA(imgs,p):
    a,b,c = imgs.shape
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w,v = np.linalg.eigh(G_t)
    w = w[::-1]
    v = v[::-1]
    for k in range(c):
        alpha = sum(w[:k])*1.0/sum(w)
        if alpha >= p:
            u = v[:,:k]
            break
    return u

a=pd.read_csv("1.csv")
print(a[1])