import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
import warnings
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric
import networkx as nx
import timeit
from sklearn.cluster import KMeans
import math
from distances import MMFD, MFD
from mfd_kd import MFD_KD
# Jicong Fan. Graph Minimum Factor Distance and Its Application to Large-Scale Graph Data Clustering. ICML 2025.
# fanjicong@cuhk.edu.cn
warnings.filterwarnings('ignore')

class graph(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

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

dataset_name = 'PROTEINS_full'
# AIDS PROTEINS_full ENZYMES REDDIT-MULTI-5K  DBLP_v1 REDDIT-MULTI-12
dataset = TUDataset('./', name=dataset_name,use_node_attr=True)
g_list=[]
for i in range(len(dataset)):
    A=torch_geometric.utils.to_dense_adj(dataset[i].edge_index)
    A=torch.squeeze(A)
    g=graph()
    g.adj_mat_numpy_array=A.numpy()
    g.X=dataset[i].x.numpy()
    if A.shape[0]-g.X.shape[0] !=0:
        # print(torch.sort(torch.reshape(dataset[i].edge_index,[-1,])),g.X.shape[0])
        print(dataset[i].edge_index.shape, A.shape[0],g.X.shape[0])
    g.label=np.squeeze(dataset[i].y.numpy())
    g_list.append(g)   

n_classes=2
if dataset_name == 'ENZYMES':
    n_classes=6
if dataset_name == 'REDDIT-MULTI-5K':
    n_classes=5
if dataset_name == 'REDDIT-MULTI-12K':
    n_classes=11

ground_truth = np.array([g.label for g in g_list])
N=len(g_list)

cluster_alg='MMFD-KM'   #### MMFD MMFD-KM MFD MFD-KD  # use low-rank by default
n_trials=1
nmi=np.zeros(n_trials)
acc=np.zeros(n_trials)
ari=np.zeros(n_trials)

if cluster_alg=='MMFD':
    g,D=MMFD(g_list,low_rank=True,d=20,use_attr=False,attr_beta=1)
    c=5 # c=3 for ENZYMES
    gamma=1/(c*np.sum(np.abs(D))/N/(N-1))**2
    K=np.exp(-gamma*D**2)
    
if cluster_alg=='MFD':
    D=MFD(g_list,d=30,beta=0.01,max_iter=10,use_attr=False,attr_beta=1)#
    c=5 # c=3 for ENZYMES
    gamma=1/(c*np.sum(np.abs(D))/N/(N-1))**2
    K=np.exp(-gamma*D**2)
    
if cluster_alg=='MMFD' or cluster_alg=='MFD':
    print('Spectral clustering......')
    for i in range(n_trials):
        print('Trials ',i+1)
        sc = SpectralClustering(n_clusters=n_classes, affinity='precomputed', n_init=1,eigen_tol=1e-6)
        sc_pred = sc.fit(K)
        y_pred = sc_pred.labels_
        nmi[i] = normalized_mutual_info_score(ground_truth, y_pred)
        acc[i] = acc_hungarian(ground_truth, y_pred)
        ari[i] = adjusted_rand_score(ground_truth, y_pred)

if cluster_alg=='MMFD-KM':
    g,D=MMFD(g_list,low_rank=True,d=20,use_attr=False,attr_beta=1)
    print('Conducting MMFD-KM......')
    for i in range(n_trials):
        kmeans = KMeans(n_clusters=n_classes,n_init=1).fit(g)  
        y_pred=kmeans.labels_
        nmi[i] = normalized_mutual_info_score(ground_truth, y_pred)
        acc[i] = acc_hungarian(ground_truth, y_pred)
        ari[i] = adjusted_rand_score(ground_truth, y_pred)

if cluster_alg=='MFD-KD':
    for i in range(n_trials):
        y_pred,dist=MFD_KD(g_list,K=n_classes,d=30,beta=0.1,use_attr=False,attr_beta=1,use_kmeans=True,y_true=ground_truth,iter_R=5,iter_C=5,iter_outer=20)
        # aids d=30, beta=0.01,iter_r=iter_c=5,iter_outer=20
        # proteins d=30, beta=0.1,iter_r=iter_c=5,iter_outer=20
        # Enz d=30, beta=0.1,iter_r=iter_c=5,iter_outer=20
        # Reddit5 d=30, beta=0.1,iter_r=iter_c=5,iter_outer=10 
        # DBLP_v1 d=30, beta=1,iter_r=iter_c=5,iter_outer=20 
        # Reddit11 d=30, beta=0.1,iter_r=iter_c=5,iter_outer=10
        print(dist)
        kmeans = KMeans(n_clusters=n_classes,n_init=1).fit(dist)  
        y_pred=kmeans.labels_
        nmi[i] = normalized_mutual_info_score(ground_truth, y_pred)
        acc[i] = acc_hungarian(ground_truth, y_pred)
        ari[i] = adjusted_rand_score(ground_truth, y_pred)
        
print('acc:',np.mean(acc),'+-',np.std(acc))
print('nmi:',np.mean(nmi),'+-',np.std(nmi))      
print('ari:',np.mean(ari),'+-',np.std(ari))



