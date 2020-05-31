import numpy as np
import pandas as pd
from sklearn import svm
import scipy.sparse as sp


def Knn(S,n):
    mat = np.zeros([imgs.shape[0], imgs.shape[0]], dtype=np.int32)
    aa = np.argsort(S)
    for i in range(S.shape[0]):
        for r in aa[i,0:n]:
            mat[i, r] = S[i,r]
    return np.maximum(mat,mat.T)

def dijkstra(A):
    nn=A.shape[0]
    D=np.zeros([nn,nn])
    for i in range(nn):
        Di=np.Inf*np.ones(nn-i,dtype=np.int32)
        Di[0]=0
        NdLb=np.arange(nn-i)
        NotOut=np.ones(nn-i, dtype=bool)
        for j in range(nn-i):
            ind=np.argmin(Di[NotOut])
            jj=NdLb[ind]
            Dj = Di[jj]
            NotOut[jj]=False
            NdLb=np.delete(NdLb, ind)
            Aj=np.where(A[jj, :] > 0)[0]
            for k in Aj:
                Dk=Dj+A[jj,k]
                Di[k]=min(Di[k],Dk)
        D[i,i:nn]=Di
        if i>0:
            D[i, i:nn] = np.min(np.tile(D[:i+1, i], [nn - i, 1]).T + D[:i+1, i:nn], 0)
        A=np.delete(np.delete(np.array(A), 0, 0).T, 0, 0).T
    return D



if __name__ == "__main__":
    NN=30
    a = pd.read_csv("./mix.csv")
    a = a.drop(a.columns[0], 1)
    imgs = a.to_numpy()
    imgs = np.array(imgs > 128, dtype=np.int0)
    D2 = np.tile(sum(imgs.T * imgs.T), [imgs.shape[0],1])
    S=D2+D2.T-2*np.dot(imgs,imgs.T)
    result = pd.DataFrame()
    for kk in range(15):
        A=Knn(S,35+kk*2)
        S1=dijkstra(np.sqrt(A))
        N=S.shape[0]
        S2=np.power(S1, 2)+np.power(S1, 2).T
        S2=S2
        G = -.5 * (S2 - np.dot(sum(S2), np.ones(N)) / N - np.dot(np.ones(N), sum(S2)) / N + np.sum(S2) / (N * N))
        w, v = np.linalg.eigh(G)
        ds=pd.DataFrame(v[:, N-NN:N])
        ds.to_csv("isomap.csv")
        yy = pd.read_csv("2.csv")
        y = yy[yy.columns[0]]
        for k in range(1, 20):
            X = ds.loc[0:y.shape[0]-1,NN-k:NN]
            test = ds.loc[y.shape[0]:,NN-k:NN]
            clf = svm.SVC()
            clf.fit(X, y)
            result[kk*20+k] = clf.predict(test)
    result.to_csv("./test/isomap.csv")