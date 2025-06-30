import numpy as np
import math
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_adjacency_matrix(adj):
    adj_ = sp.csr_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    adj_m1 = adj_ - sp.dia_matrix((adj_.diagonal()[np.newaxis, :], [0]), shape=adj_.shape)
    adj_m1.eliminate_zeros()
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        'adj_norm': adj_normalized,
        'adj_label': adj_,
        'norm_value': norm_m1
    }

    return graph_dict


def graph_construction_scRNA(adata, n_comps=50, n_neighbors=20):

    np.random.seed(42)
    n_comps = min(50, adata.shape[1] - 1)  # 确保 n_comps 不超过基因数减一
    print("==> SC dataset Cell numbers: %d"%adata.shape[0])
    adata.X = adata.X.astype('float32')
    sc.tl.pca(adata, n_comps=n_comps)

    cs_matrix = cosine_similarity(adata.obsm['X_pca'], adata.obsm['X_pca'])
    top_K = np.argsort(cs_matrix, axis=1)[:,-n_neighbors:]
    mask = np.zeros_like(cs_matrix)

    for i in range(adata.shape[0]):
        mask[i][top_K[i]] = 1

    return preprocess_adjacency_matrix(np.multiply(cs_matrix, mask))


def spatial_weight(arr, alpha, mu, sigma):
    results = alpha*np.exp(-(arr-mu)**2/2/(sigma**2))
    return results


def graph_construction_spatial(adata, n_comps=50, n_neighbors=20, dis_sigma=50, delta=2000):
    np.random.seed(42)
    n_comps = min(50, adata.shape[1] - 1)  # 确保 n_comps 不超过基因数减一
    print("==> ST dataset Cell numbers: %d"%adata.shape[0])
    adata.X = adata.X.astype('float32')
    sc.tl.pca(adata, n_comps=n_comps)

    parts = math.ceil(adata.shape[0] / delta)

    comprehensive_weight = np.zeros((adata.shape[0], adata.shape[0]),dtype=np.float16)

    for i in range(parts):
        cs_matrix = cosine_similarity(adata.obsm['X_pca'][i*delta:(i+1)*delta], adata.obsm['X_pca'])
        euc_distance = distance.cdist(adata.obsm["spatial"][i*delta:(i+1)*delta], adata.obsm["spatial"])
        sw = spatial_weight(euc_distance, alpha=1, mu=0, sigma=dis_sigma)  # sigma
        weight = np.multiply(cs_matrix, sw)
        top_K = np.argsort(weight, axis=1)[:,-n_neighbors:]
        mask = np.zeros_like(weight)

        for k in range(mask.shape[0]):
            mask[k][top_K[k]] = 1
        comprehensive_weight[i*delta:(i+1)*delta] = np.multiply(weight, mask)
        print('[Num %3d/%3d] \r'% (i+1, parts), end='')

    return preprocess_adjacency_matrix(comprehensive_weight)