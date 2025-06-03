import numpy as np
from scipy.spatial import distance_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import pairwise_distances
import math
# Jicong Fan. Graph Minimum Factor Distance and Its Application to Large-Scale Graph Data Clustering. ICML 2025.
# fanjicong@cuhk.edu.cn


def combine_graph(A1,A2):
    n1=A1.shape[0]
    n2=A2.shape[0]
    if n1 != n2:
        print('Size not match:','n1=',n1,'n2=',n2)
        # raise Exception("The two input graphs should have the same size")
    A=np.zeros((n1+n2,n1+n2))
    A[0:n1,0:n1]=A1
    A[n1:n1+n2,n1:n1+n2]=A2
    if n1==n2:
        A[0:n1,n1:n1+n1]=np.eye(n1)*0.5
        A[n1:n1+n1,0:n1]=np.eye(n1)*0.5
    return A

def graph_aug_attr(g_list,attr_beta=1):
    N=len(g_list)
    for i in range(N):
        if i==0:
            X_all=g_list[i].X
        else:
            X_all=np.concatenate((X_all,g_list[i].X),axis=0)          
    x_mean=np.mean(X_all,axis=0)
    x_std=np.std(X_all,axis=0)+1e-10
    mean_dist=np.zeros(N)
    for i in range(N):
        g_list[i].X=(g_list[i].X- x_mean)/(x_std)
        g_list[i].D=distance_matrix(g_list[i].X,g_list[i].X,p=1)
        n=g_list[i].X.shape[0]
        mean_dist[i]=np.sum(g_list[i].D)/n/(n-1)
    thr=attr_beta*np.mean(mean_dist) # beta: AIDS 0.01 PROTEIN 1 ENZ 1 or 0.1
    gamma=1/thr**2
    for i in range(N):
        n=g_list[i].X.shape[0]
        g_list[i].A_attr=np.exp(-gamma*(g_list[i].D)**2)*(1-np.eye(g_list[i].D.shape[0]))
        g_list[i].adj_mat_numpy_array=combine_graph(g_list[i].adj_mat_numpy_array,g_list[i].A_attr)
    return g_list

def MMFD(g_list,low_rank=True,d=10,use_attr=False,attr_beta=1):
    if use_attr==True:
        g_list=graph_aug_attr(g_list,attr_beta) 
    N=len(g_list)
    g=np.zeros((N,1))
    cc=math.floor(N/10)
    for i in range(N):     
        if i % cc ==0:
            print('Computing SVD:',i,' of ',N)
        n=g_list[i].adj_mat_numpy_array.shape[1]
        A=g_list[i].adj_mat_numpy_array+np.eye(n)
        if low_rank==True:
            if n<100:
                S,V = np.linalg.eig(A.astype(np.float32))
                S=np.abs(S)
                id=np.argsort(-S)
                S=S[id]
                VT=V[:,id].T
            else:
                U,S,VT = randomized_svd(A.astype(np.float32), n_components=d, random_state=0)
            Z=VT.T[:,0:d]@np.diagflat(S[0:d]**0.5)
            if Z.shape[1]<d:
                Zt=np.zeros((Z.shape[0],d))
                Zt[:,0:Z.shape[1]]=Z
                Z=Zt
        else:
            S,V = np.linalg.eig(A.astype(np.float32))
            if min(S)>=0:
                print('Find one PSD matrix!')
            S=np.abs(S)
            id=np.argsort(-S)
            S=S[id]
            VT=V[:,id].T
            Z=VT.T@np.diagflat(S**0.5)
        g[i,:]=np.sqrt(np.sum(Z@Z.T))/n   
    D=np.absolute(g-g.T)
    return g,D

    
def MFD(g_list,d=10,beta=1,max_iter=10,use_attr=False,attr_beta=1):
    if use_attr==True:
        g_list=graph_aug_attr(g_list,attr_beta)
    N=len(g_list)         
    n_nodes=np.zeros(N)
    mDZ=np.zeros(N)
    for i in range(N):     
        if i % 200 ==0:
            print('Computing SVD:',i,' of ',N)
        n=g_list[i].adj_mat_numpy_array.shape[1]
        A=g_list[i].adj_mat_numpy_array+np.eye(n)
        if n<100:
            S,V = np.linalg.eig(A.astype(np.float32))
            S=np.abs(S)
            id=np.argsort(-S)
            S=S[id]
            VT=V[:,id].T
        else:
            U,S,VT = randomized_svd(A.astype(np.float32), n_components=d, random_state=0)
        Z=VT.T[:,0:d]@np.diagflat(S[0:d]**0.5)
        if Z.shape[1]<d:
            Zt=np.zeros((Z.shape[0],d))
            Zt[:,0:Z.shape[1]]=Z
            Z=Zt
        Z=np.real(Z)
        n_nodes[i]=n
        Ah=Z@Z.T
        g_list[i].Z=Z  
        dA=np.diagonal(Ah).reshape(-1, 1)
        g_list[i].dA=dA
        mA=np.sum(np.exp(-beta*(dA+dA.T-2*Ah)))/n/n
        g_list[i].mA=mA
        mDZ[i]=np.mean(pairwise_distances(Z,metric='euclidean'))     
    D=np.zeros((N,N))
    for i in range(N):
        if i % 100 ==0:
            print('Computing MFD:',i,' of ',N)
        for j in range(i+1,N):
            D[i,j]=compute_MFD(beta,max_iter,g_list[i].Z,g_list[j].Z,g_list[i].mA,g_list[j].mA,g_list[i].dA,g_list[j].dA)
    D=np.maximum(D,D.T)
    return D

def compute_MFD(beta,max_iter,Z1,Z2,mA1,mA2,dA1,dA2):
    n1,d=Z1.shape
    n2,d=Z2.shape
    R=np.eye(d)
    Phi_1=Z1.T
    Phi_2=Z2.T
    alpha=np.exp(-beta*(dA1+dA2.T))
    sum_alpha=np.sum(alpha)
    for i in range(max_iter):
        RPhi_2=R@Phi_2
        C=2*beta*Phi_1.T@RPhi_2
        w=2*beta*np.exp(C)
        aw=alpha*w
        P=Phi_1@aw@Phi_2.T+R*sum_alpha*1e-5
        U,s,VT=np.linalg.svd(P)
        R_new=U@VT
        # e=np.linalg.norm(R-R_new,'fro')/np.sqrt(d)
        m12=np.sum(np.exp(-beta*(dA1+dA2.T-2*Phi_1.T@RPhi_2)))/n1/n2
        dist=np.abs(mA1+mA2-2*m12)**0.5
        R=R_new
        # if i==0 or i==10 or i==max_iter-1:
        #     print('iter=', i, ' epsilon=',e,' dist=',dist)
    return dist