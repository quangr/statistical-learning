import numpy as np
import pandas as pd
from sklearn import svm
import twodpca

p = 4
q = 4
a = pd.read_csv("./1.csv")
a = a.drop(a.columns[0], 1)
a = twodpca.csv2np(a)
cp, u, v = twodpca.TTwoDPCA(a, p, q)
df2 = pd.DataFrame()
for index in range(cp.shape[0]):
    df2[index] = cp[index].reshape(p * q)
yy = pd.read_csv("2.csv")
y=yy[yy.columns[0]]
X=df2.T
clf = svm.SVC()
clf.fit(X, y)
test = pd.read_csv("./test/1.csv")
test = test.drop(test.columns[0], 1)
test = twodpca.csv2np(test)
aaa=(np.dot(u.T,np.dot(test[1]-a.mean(),v))[28-p:28,28-q:28].T).reshape(p*q)
print(clf.predict(pd.DataFrame(aaa).T))