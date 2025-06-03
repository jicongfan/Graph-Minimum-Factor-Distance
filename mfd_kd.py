import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
import timeit
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import math
from scipy.optimize import linear_sum_assignment
from distances import graph_aug_attr
# Jicong Fan. Graph Minimum Factor Distance and Its Application to Large-Scale Graph Data Clustering. ICML 2025.
# fanjicong@cuhk.edu.cn



def compute_kernel_matrix(X,Y,gamma):
    D2=pairwise_distances(X,Y,metric='sqeuclidean')
    K=np.exp(-gamma*D2)
    return K
    
def compute_MFD(beta,Z1,Z2,mA1,mA2,dA1,dA2,T):
    n1,d=Z1.shape
    n2,d=Z2.shape
    R=np.eye(d)
    Phi_1=Z1.T
    Phi_2=Z2.T
    alpha=np.exp(-beta*(dA1+dA2.T))
    sum_alpha=np.sum(alpha)
    for i in range(T+1):
        RPhi_2=R@Phi_2
        C=2*beta*Phi_1.T@RPhi_2
        w=2*beta*np.exp(C)
        aw=alpha*w
        P=Phi_1@aw@Phi_2.T+R*sum_alpha*1e-5
        U,s,VT=np.linalg.svd(P)
        R_new=U@VT
        e=np.linalg.norm(R-R_new,'fro')/np.sqrt(d)
        m12=np.sum(np.exp(-beta*(dA1+dA2.T-2*Phi_1.T@RPhi_2)))/n1/n2
        dist=np.abs(mA1+mA2-2*m12)**0.5
        R_old=R
        R=R_new
        # if i==0 or i==10 or i==T-1:
        #     print('iter=', i, ' epsilon=',e,' dist=',dist)
    return dist,R_old

def compute_gradient(Z,R,C,beta):
    n_i=Z.shape[0]
    n=C.shape[0]
    K=compute_kernel_matrix(Z,C@R.T,beta)
    g1=4*beta/n/n_i*np.diagflat(np.sum(K,axis=0))
    g2=-4*beta/n/n_i*R.T@Z.T@K
    loss_partial=-2/n_i/n*np.sum(K)
    return g1,g2.T,loss_partial
    
    
def initialize_C(K,n_bar,d,Z_all,Z_mean,use_kmeans=True):
    C=np.zeros((K,n_bar,d))
    if use_kmeans==True:
        kmeans = KMeans(n_clusters=K,n_init=10).fit(Z_mean)  
        C0=kmeans.cluster_centers_
        for i in range(K):
            # C[i,:,:]=C0[i,:]+np.random.randn(n_bar,d)@np.diagflat(np.std(Z_all,axis=0))
            temp=np.random.uniform(-1,1,(n_bar,d))
            C[i,:,:]=C0[i,:]+temp/np.std(temp,axis=0)@np.diagflat(np.std(Z_all,axis=0))*1
    else:  
        ids=np.random.permutation(range(Z_all.shape[0]))
        C=Z_all[ids[0:n_bar*K],:]
        C=np.reshape(C,(K,n_bar,d))+np.random.randn(K,n_bar,d)@np.diagflat(np.std(Z_all,axis=0))*0.5
        
    return C 
  
def update_label(g_list,C,beta,iter_R):
    K=C.shape[0]
    n_bar=C.shape[1]
    N=len(g_list)
    dist=np.zeros((N,K))
    mB=np.zeros(K)
    dB=np.zeros((n_bar,K))
    for j in range(K):
        Bh=C[j,:,:]@C[j,:,:].T
        dB[:,j]=np.diagonal(Bh)
        mB[j]=np.sum(np.exp(-beta*(dB[:,j]+dB[:,j].T-2*Bh)))/n_bar/n_bar
    for i in range(N):
        g_list[i].R_list=[]
        for j in range(K):
            dist[i,j],R = compute_MFD(beta,g_list[i].Z,C[j,:,:],g_list[i].mA,mB[j],g_list[i].dA,dB[:,j],iter_R)
            g_list[i].R_list.append(R)
    Y=np.argmin(dist,axis=1) 
    return Y, g_list, dist

def update_center(g_list,C,Y,beta,iter_C):
    K,n_bar,d=C.shape
    loss_all=0
    for j in range(K):
        g_index_j=np.where(Y==j)[0]
        loss_t=0
        for t in range(iter_C):
            Kcc=compute_kernel_matrix(C[j,:,:],C[j,:,:],beta)
            g0=4*beta/n_bar/n_bar*(Kcc-np.diagflat(np.sum(Kcc,axis=0)))
            loss_t=0
            g1=np.zeros((n_bar,n_bar))
            g2=np.zeros((n_bar,d))
            for i in g_index_j:
                g1_i,g2_i,loss_i=compute_gradient(g_list[i].Z,g_list[i].R_list[j],C[j,:,:],beta)
                g1=g1+g1_i
                g2=g2+g2_i
                loss_t=loss_t+loss_i
            loss_t=loss_t+np.mean(Kcc)*len(g_index_j)
            # print(np.sum(g0),len(g_index_j),np.sum(g1))
            E=g0*len(g_index_j)+g1
            gC=E.T@C[j,:,:]+g2
            U,lambda_max,VT = randomized_svd(E.astype(np.float32), n_components=1, random_state=0)
            lr=0.5/lambda_max
            if len(g_index_j)>0:
                C[j,:,:]=C[j,:,:]-lr*gC
        loss_all=loss_all+loss_t
    return C,loss_all

def prepare_MFD(g_list,K,d=5,beta=1,use_attr=False,attr_beta=1,use_kmeans=True):
    if use_attr==True:
        g_list=graph_aug_attr(g_list,attr_beta) 
    N=len(g_list)         
    Z_mean=np.zeros((N,d))
    n_nodes=np.zeros(N)
    mDZ=np.zeros(N)
    loss_0=0
    cc=math.floor(N/10)
    for i in range(N):     
        if i % cc ==0:
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
        Z_mean[i,:]=np.mean(Z,axis=0) 
        n_nodes[i]=Z.shape[0]
        if i == 0:
            Z_all=Z
        else:
            Z_all=np.concatenate((Z_all,Z),axis=0)
        Ah=Z@Z.T
        g_list[i].Z=Z  
        dA=np.diagonal(Ah).reshape(-1, 1)
        g_list[i].dA=dA
        g_list[i].K=np.exp(-beta*(dA+dA.T-2*Ah))
        g_list[i].mA=np.sum(g_list[i].K)/n/n
        loss_0=loss_0+g_list[i].mA
        mDZ[i]=np.mean(pairwise_distances(Z,metric='euclidean'))
    n_bar=np.minimum(int(np.median(n_nodes)),50)
    print('The size of each center is',n_bar,'x',d)
    C=initialize_C(K,n_bar,d,Z_all,Z_mean,use_kmeans=True) 
    return g_list,C,loss_0  
 
def MFD_KD(g_list,K=2,d=5,beta=1,use_attr=False,attr_beta=1,use_kmeans=True,y_true=[],iter_R=10,iter_C=1,iter_outer=10):
    g_list,C,loss_0=prepare_MFD(g_list,K,d=d,beta=beta,use_attr=use_attr,attr_beta=attr_beta,use_kmeans=use_kmeans)
    for iter in range(iter_outer):
        Y, g_list,dist = update_label(g_list,C,beta,iter_R)
        C,loss = update_center(g_list,C,Y,beta,iter_C)
        if iter % 1 ==0:
            if len(y_true)==0:
                print('Iteration:',iter,' Loss=',loss+loss_0,' cluster sizes:', np.unique(np.bincount(Y)))
            else:
                acc,nmi,ari=evaluate(y_true,Y)
                print('Iteration:',iter,' Loss=',loss+loss_0,' cluster sizes:', np.unique(np.bincount(Y)))
                print('acc:',acc,' nmi:',nmi,' ari:',ari)
        loss_old=loss
    return Y,dist

def evaluate(ground_truth,y_pred):
    acc = acc_hungarian(ground_truth, y_pred)
    nmi = normalized_mutual_info_score(ground_truth, y_pred)
    ari = adjusted_rand_score(ground_truth, y_pred)
    return acc,nmi,ari


def acc_hungarian(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        
