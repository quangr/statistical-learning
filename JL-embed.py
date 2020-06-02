import numpy as np
import pandas as pd
from sklearn import svm
import twodpca


def random_projection(imgs, p):
    b = sum(imgs)/imgs.shape[0]
    A = np.random.normal(size=[imgs[0].shape[0],p])
    cp = np.empty(imgs.shape[0], np.ndarray)
    for index in range(imgs.shape[0]):
        cp[index] = np.dot(imgs[index] - b, A)
    return cp,A

a = pd.read_csv("./1.csv")
a = a.drop(a.columns[0], 1)
imgs = a.to_numpy()
imgs=np.array(imgs>0,dtype=np.double)
yy = pd.read_csv("2.csv")
y=yy[yy.columns[0]]
test = pd.read_csv("./test/1.csv")
test = test.drop(test.columns[0], 1)
test = test.to_numpy()
test=np.array(test>0,dtype=np.double)
result = pd.DataFrame()
for i in range(test.shape[0]):
    test[i] = test[i] - imgs.mean()
for k in range(1,10):
    aaa = []
    cp, u = random_projection(imgs,25)
    for i in range(test.shape[0]):
        aaa.append(np.dot(test[i], u))
    aa = pd.DataFrame(aaa)
    df2 = pd.DataFrame()
    for index in range(cp.shape[0]):
        df2[index] = cp[index]
    X=df2.T
    clf = svm.SVC()
    clf.fit(X, y)
    result[k]=clf.predict(aa)
result.to_csv("./test/random.csv")