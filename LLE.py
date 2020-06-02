import numpy as np
import pandas as pd
from sklearn import svm
from scipy import ndimage
import scipy.sparse as sp


def Knn(S,n):
    mat = np.zeros([S.shape[0], S.shape[0]], dtype=np.double)
    aa = np.argsort(S)
    for i in range(S.shape[0]):
        for r in aa[i,0:n]:
            mat[i, r] = S[i,r]
    return np.maximum(mat,mat.T)

def LLE(X,A):
    N=X.shape[0]
    W=np.zeros([N,N])
    for i in range(N):
        z=X[(A>0)[i,:]]-X[i]
        C=np.dot(z,z.T)
        W[i,(A>0)[i,:]]=np.linalg.solve(C+np.eye(C.shape[0],C.shape[0])*(C.shape[0]>np.linalg.matrix_rank(C))*0.03, np.ones(C.shape[0]))
    W=(W.T/np.sum(W,1)).T
    return W

def allLLE(X,t,Nb):
    NN=35
    D2 = np.tile(sum(X.T * X.T), [X.shape[0],1])
    S=D2+D2.T-2*np.dot(X,X.T)
    result = pd.DataFrame()
    N=X.shape[0]
    A = Knn(S, Nb)
    W=LLE(X,A>0)
    yy = pd.read_csv("2.csv")
    y = yy[yy.columns[0]]
    for i in range(y.shape[0]):
        A1=A>0
        A1[i, np.concatenate([y == y[i], np.zeros(A.shape[0] - y.shape[0], dtype=bool)])]=True
    W1=LLE(X,A1)
    W=t*W1+(1-t)*W
    K=np.dot((np.eye(N,N)-W).T,(np.eye(N,N)-W))
    w, v = np.linalg.eigh(K)
    ds = pd.DataFrame(v[:, 1:NN])
    ds.to_csv("LLE.csv")
    for k in range(1, 32):
         XX = ds.loc[0:y.shape[0]-1,0:k]
         test = ds.loc[y.shape[0]:,0:k]
         clf = svm.SVC()
         clf.fit(XX, y)
         result[k] = clf.predict(test)
    result.to_csv("./test/LLE.csv")

def sinLLE1(X,t,Nb):
    NN=40
    yy = pd.read_csv("2.csv")
    y = yy[yy.columns[0]]
    nX=np.zeros([X.shape[0]+10,X.shape[1]])
    nX[:y.shape[0],] = X[:y.shape[0],]
    nX[y.shape[0] + 10:,] = X[y.shape[0]:,]
    y=y.append(pd.Series(np.ones(5,dtype=np.int64)), ignore_index=True)
    y=y.append(pd.Series(9*np.ones(5,dtype=np.int64)), ignore_index=True)
    for g in range(5):
        nX[y.shape[0]+g,:]=ndimage.rotate(imgs[52].reshape([26, 26]), (g-2)*10, reshape=False).reshape(26*26)
    for g in range(5):
        nX[y.shape[0]+5+g,:]=ndimage.rotate(imgs[46].reshape([26, 26]), (g-2)*10, reshape=False).reshape(26*26)
    X=nX
    result = pd.DataFrame()
    for kk in range(X.shape[0]-y.shape[0]):
        sl=np.zeros(X.shape[0],dtype=bool)
        sl[:y.shape[0]]=True
        sl[kk+y.shape[0]]=True
        Xt=X[sl]
        D2 = np.tile(sum(Xt.T * Xt.T), [Xt.shape[0], 1])
        S = D2 + D2.T - 2 * np.dot(Xt, Xt.T)
        N = Xt.shape[0]
        A = Knn(S, Nb)
        W = LLE(Xt, A > 0)
        for i in range(y.shape[0]):
            A1=A>0
            A1[i, np.concatenate([y == y[i], np.zeros(A.shape[0] - y.shape[0], dtype=bool)])]=True
            if(y[i]==1|y[i]==9):
                A1[i, np.concatenate([y != y[i], np.zeros(A.shape[0] - y.shape[0], dtype=bool)])] = False
        W1=LLE(Xt,A1)
        W=t*W1+(1-t)*W
        K=np.dot((np.eye(N,N)-W).T,(np.eye(N,N)-W))
        w, v = np.linalg.eigh(K)
        ds = pd.DataFrame(v[:, 1:NN])
        for k in range(1, 38):
             XX = ds.loc[0:y.shape[0]-1,0:k]
             test = ds.loc[y.shape[0]:,0:k]
             clf = svm.SVC()
             clf.fit(XX, y)
             result.loc[kk,k] = clf.predict(test)[0]
    result.to_csv("./test/LLE.csv")
def sinLLE(X,t,Nb):
    NN=40
    yy = pd.read_csv("2.csv")
    y = yy[yy.columns[0]]
    nX=np.zeros([X.shape[0]+10,X.shape[1]])
    nX[:y.shape[0],] = X[:y.shape[0],]
    nX[y.shape[0] + 10:,] = X[y.shape[0]:,]
    result = pd.DataFrame()
    for kk in range(X.shape[0]-y.shape[0]):
        sl=np.zeros(X.shape[0],dtype=bool)
        sl[:y.shape[0]]=True
        sl[kk+y.shape[0]]=True
        Xt=X[sl]
        D2 = np.tile(sum(Xt.T * Xt.T), [Xt.shape[0], 1])
        S = D2 + D2.T - 2 * np.dot(Xt, Xt.T)
        N = Xt.shape[0]
        A = Knn(S, Nb)
        W = LLE(Xt, A > 0)
        for i in range(y.shape[0]):
            A1=A>0
            A1[i, np.concatenate([y == y[i], np.zeros(A.shape[0] - y.shape[0], dtype=bool)])]=True
            # if(y[i]==1|y[i]==9):
            #     A1[i, np.concatenate([y != y[i], np.zeros(A.shape[0] - y.shape[0], dtype=bool)])] = False
        W1=LLE(Xt,A1)
        W=t*W1+(1-t)*W
        K=np.dot((np.eye(N,N)-W).T,(np.eye(N,N)-W))
        w, v = np.linalg.eigh(K)
        ds = pd.DataFrame(v[:, 1:NN])
        for k in range(20, 38):
             XX = ds.loc[0:y.shape[0]-1,0:k]
             test = ds.loc[y.shape[0]:,0:k]
             clf = svm.SVC()
             clf.fit(XX, y)
             result.loc[kk,k] = clf.predict(test)[0]
    result.to_csv("./test/LLE.csv")


if __name__ == "__main__":
    a = pd.read_csv("./mix.csv")
    a = a.drop(a.columns[0], 1)
    imgs = a.to_numpy()
    imgs = np.array(imgs > 0, dtype=np.double)
    allLLE(imgs,0.5,90)
    #sinLLE(imgs,0.5,8)
