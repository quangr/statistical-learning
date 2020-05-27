import os
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import data
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_local
from PIL import Image
import operator
from skimage import img_as_ubyte


def csv2np(imgs):
    imgs = imgs.to_numpy()
    nimgs = np.empty(imgs.shape[0], np.ndarray)
    for index in range(imgs.shape[0]):
        nimgs[index] = imgs[index].reshape(28, 28)
    return nimgs


def TwoDPCA(imgs, p):
    b = imgs.mean()
    A = np.zeros([max(imgs[0].shape), max(imgs[0].shape)])
    for index in range(imgs.shape[0]):
        temp = imgs[index] - b
        A = A + np.dot(temp.T, temp)
    A = A / (imgs.shape[0])
    w, v = np.linalg.eigh(A)
    cp = np.empty(imgs.shape[0], np.ndarray)
    for index in range(imgs.shape[0]):
        cp[index] = np.dot(imgs[index] - b, v)[:, max(imgs[0].shape) - p:max(imgs.shape) - 1]
    return cp, v


def TTwoDPCA(imgs, p, q):
    cp, v = TwoDPCA(imgs, p)
    for i in range(cp.shape[0]):
        cp[i] = cp[i].T
    cp1, u = TwoDPCA(cp, q)
    return cp1, u, v


def rebuildimg(cp, u, v, b):
    rimg = np.empty(cp.shape[0], np.ndarray)
    n = u.shape[0]
    p, q = cp[0].shape
    for index in range(rimg.shape[0]):
        temp = np.zeros([p, n])
        temp[:, n - q:n] = cp[index]
        temp = np.dot(temp, u.T)
        rimg[index] = np.zeros(u.shape)
        rimg[index][:, n - p:n] = temp.T
        rimg[index] = np.dot(rimg[index], v.T) + b
        rimg[index] = np.round(np.maximum(rimg[index], 0))
    return rimg


p = 7
q = 7
a = pd.read_csv("1.csv")
a = a.drop(a.columns[0], 1)
a = csv2np(a)
cp, u, v = TTwoDPCA(a, p, q)
imgs = rebuildimg(cp, u, v, a.mean())
# io.imshow(imgs[1].astype(np.uint8), cmap=cm.gray)
# plt.show()

df2 = pd.DataFrame()
for index in range(cp.shape[0]):
    df2[index] = cp[index].reshape(p * q)
(df2.T).to_csv("2.csv")
