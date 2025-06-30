import time

from Options import TrainOptions
from model import spaAdaptModel
from Graph import graph_construction_scRNA, graph_construction_spatial
from utils import sparse_mx_to_torch_sparse_tensor
import pandas as pd
import numpy as np
import scanpy as sc
import json
import torch
import scipy.sparse

def neighbor_nodes(sparse_matrix, nodes_idx):
    return np.nonzero(sparse_matrix[nodes_idx].sum(axis=0))[1]

def retrieve_subgraph(graph_dict, expression_tensor, label_tensor, nodes_idx):
    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'][nodes_idx,:][:,nodes_idx]) #将一个 SciPy 稀疏矩阵转换为 PyTorch 的稀疏张量
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'][nodes_idx,:][:,nodes_idx])
    subgraph_dict['norm_value'] = graph_dict['norm_value']
    sub_expression_tensor = expression_tensor[nodes_idx]
    sub_label_tensor = label_tensor[nodes_idx]
    return subgraph_dict, sub_expression_tensor, sub_label_tensor


def retrieve_fixed_size_subgraph_with_target(graph_dict, expression_tensor, label_tensor, target_nodes_idx, fixed_size=5000):
    first_neighbors = neighbor_nodes(graph_dict['adj_norm'], target_nodes_idx)

    second_neighbors = neighbor_nodes(graph_dict['adj_norm'], first_neighbors)

    all_related_nodes = np.unique(np.concatenate((target_nodes_idx, first_neighbors, second_neighbors)))

    remaining_size = fixed_size - len(target_nodes_idx)
    if remaining_size > 0:
        additional_nodes = np.setdiff1d(all_related_nodes, target_nodes_idx)

        if len(additional_nodes) <= remaining_size:
            selected_additional_nodes = additional_nodes
        else:
            np.random.seed(42)
            selected_additional_nodes = np.random.choice(additional_nodes, size=remaining_size, replace=False)

        selected_nodes = np.concatenate((target_nodes_idx, selected_additional_nodes))
    else:
        selected_nodes = target_nodes_idx

    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'][selected_nodes, :][:, selected_nodes])
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'][selected_nodes, :][:, selected_nodes])
    subgraph_dict['norm_value'] = graph_dict['norm_value']

    sub_expression_tensor = expression_tensor[selected_nodes]
    sub_label_tensor = label_tensor[selected_nodes]

    return subgraph_dict, sub_expression_tensor, sub_label_tensor

def to_tensor(graph_dict):
    tensor_graph_dict = {}
    tensor_graph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'])
    tensor_graph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'])
    tensor_graph_dict['norm_value'] = graph_dict['norm_value']
    return tensor_graph_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    opt = TrainOptions().parse()

    sc_adata = sc.read(opt.sc_data)
    st_adata = sc.read(opt.st_data)
    # 去除重复基因名
    sc_adata.var_names_make_unique()

    # 找到两个数据集的共有基因
    common_genes = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    if "spatial" in st_adata.obsm:
        spatial_coords = st_adata.obsm["spatial"]  # 空间坐标
    else:
        raise ValueError("空间数据集中没有找到 'spatial' 坐标信息！")

    combined_adata = sc.concat([sc_adata, st_adata], join="inner", label="batch")

    combined_adata.obs["batch"] = ["scRNA" if i < sc_adata.n_obs else "spatial" for i in range(combined_adata.n_obs)]

    sc.pp.filter_cells(combined_adata, min_genes=1)
    sc.pp.filter_genes(combined_adata, min_cells=opt.min_cells)
    sc.pp.normalize_total(combined_adata)
    sc.pp.log1p(combined_adata)

    sc.pp.highly_variable_genes(
        combined_adata,
        n_top_genes=opt.highly_variable,
        batch_key="batch",
        flavor="seurat_v3"
    )

    combined_adata = combined_adata[:, combined_adata.var['highly_variable']].copy()

    common_sc_adata = combined_adata[combined_adata.obs["batch"] == "scRNA"].copy()
    common_st_adata = combined_adata[combined_adata.obs["batch"] == "spatial"].copy()

    common_st_adata.obsm["spatial"] = spatial_coords[:common_st_adata.n_obs]

    print(f"共有高变基因数量: {len(common_sc_adata.var_names)}")



    graph_dict_sc = graph_construction_scRNA(common_sc_adata, n_neighbors=opt.sc_neighbors)

    if scipy.sparse.issparse(common_sc_adata.X):
        x_sc = torch.Tensor(common_sc_adata.X.toarray())
    else:
        x_sc = torch.Tensor(common_sc_adata.X)
    l_sc = torch.Tensor(common_sc_adata.obs.labels)
    tensor_graph_dict_sc = to_tensor(graph_dict_sc)
    types_dict = dict(set(zip(common_sc_adata.obs.labels.to_list(), common_sc_adata.obs.cell_types.to_list())))

    graph_dict_st = graph_construction_spatial(common_st_adata, dis_sigma=opt.dis_sigma, n_neighbors=opt.st_neighbors)
    if scipy.sparse.issparse(common_st_adata.X):
        x_st = torch.Tensor(common_st_adata.X.toarray())
    else:
        x_st = torch.Tensor(common_st_adata.X)
    l_st = torch.Tensor(common_st_adata.obs.labels)
    tensor_graph_dict_st = to_tensor(graph_dict_st)

    sc_cells_number, st_cells_number = common_sc_adata.shape[0], common_st_adata.shape[0]
    print('==> The number of cells in sc dataset: %d | in st dataset: %d' % (sc_cells_number, st_cells_number))
    sc_batches = int(sc_cells_number / opt.cells_per_batch)
    st_batches = int(st_cells_number / opt.cells_per_batch)
    batches = max(sc_batches, st_batches)
    print('==> Epochs: %d, Batches: %d, sc_batches: %d, st_batches: %d' %(opt.n_epochs, batches, sc_batches, st_batches))

    best = 0
    setup_seed(1)
    model = spaAdaptModel(opt)
    model.setup(opt)

    for epoch in range(1, opt.n_epochs + 1):

        sc_cell_idxes = np.arange(sc_cells_number)
        st_cell_idxes = np.arange(st_cells_number)
        np.random.shuffle(sc_cell_idxes)
        np.random.shuffle(st_cell_idxes)

        for batch in range(batches):
            scb = batch % sc_batches
            stb = batch % st_batches

            scb_cell_idx = sc_cell_idxes[scb * opt.cells_per_batch:(scb + 1) * opt.cells_per_batch]
            stb_cell_idx = st_cell_idxes[stb * opt.cells_per_batch:(stb + 1) * opt.cells_per_batch]

            subgraph_dict_sc, subx_sc, subl_sc = retrieve_fixed_size_subgraph_with_target(graph_dict_sc, x_sc, l_sc, scb_cell_idx, fixed_size=opt.neighbor_per_batch)
            subgraph_dict_st, subx_st, subl_st = retrieve_fixed_size_subgraph_with_target(graph_dict_st, x_st, l_st, stb_cell_idx, fixed_size=opt.neighbor_per_batch)

            model.optimize_parameters(subx_sc, subl_sc, subgraph_dict_sc, subx_st, subgraph_dict_st,epoch)


            if batch % opt.print_freq == 0:
                losses = model.get_current_losses()
                print('==> epoch: %d, Batch:[%d/%d] | Loss: cls %.4f, mmd %.4f, GS %.4f, GT %.4f, CA %.4f, DIFF %.4f'%(epoch, batch, batches,
                           losses['CLS'], losses['MMD'], losses['GS'], losses['GT'], losses['CA'], losses['DIFF']))

        model.update_learning_rate()

        if epoch > 15:
            print('\n===> Start inferencing at epoch %d...'%epoch)
            results, embeddings,p_embedding = model.inference(x_st, tensor_graph_dict_st)
            results = results.cpu().numpy()
            gt_arr = np.array(common_st_adata.obs.labels)
            predictions = [int(i) for i in np.argmax(results, axis=1)]
            acc = sum(gt_arr==np.array(predictions))*1./common_st_adata.shape[0]
            print("Acc for prediction:", acc)

            if acc > best:
                best = acc
                best_results = results
                best_predictions = predictions
                embeddings = embeddings.cpu().numpy()
                p_embedding = p_embedding.cpu().numpy()
                best_embeddings = embeddings
                best_p_embeddings = p_embedding
                print('\t### Found best model at epoch %d with accuracy: %.2f'%(epoch, acc))

    print("Best acc for prediction:", best)

    cell_types = [types_dict[pred] for pred in best_predictions]

    result_adata = common_st_adata.copy()
    res_sc_adata = common_sc_adata.copy()

    result_adata.obs['predicted_label'] = best_predictions
    result_adata.obs['predicted_cell_type'] = cell_types
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    result_adata.write(f'/home/lsy/code/SpaAdapt/SpaAdapt/annotation/hpr.h5ad')



