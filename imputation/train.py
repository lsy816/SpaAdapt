import time

# from MOP.Options import TestOptions
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
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr


def neighbor_nodes(sparse_matrix, nodes_idx):
    return np.nonzero(sparse_matrix[nodes_idx].sum(axis=0))[1]


def retrieve_subgraph(graph_dict, expression_tensor, label_tensor, nodes_idx):
    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(
        graph_dict['adj_norm'][nodes_idx, :][:, nodes_idx])
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'][nodes_idx, :][:, nodes_idx])
    subgraph_dict['norm_value'] = graph_dict['norm_value']
    sub_expression_tensor = expression_tensor[nodes_idx]
    sub_label_tensor = label_tensor[nodes_idx]
    return subgraph_dict, sub_expression_tensor, sub_label_tensor


def retrieve_fixed_size_subgraph_with_target(graph_dict, expression_tensor, label_tensor, target_nodes_idx,
                                             fixed_size=5000):
    first_neighbors = neighbor_nodes(graph_dict['adj_norm'], target_nodes_idx)

    second_neighbors = neighbor_nodes(graph_dict['adj_norm'], first_neighbors)

    all_related_nodes = np.unique(np.concatenate((target_nodes_idx, first_neighbors, second_neighbors)))

    remaining_size = fixed_size - len(target_nodes_idx)
    if remaining_size > 0:
        additional_nodes = np.setdiff1d(all_related_nodes,
                                        target_nodes_idx)

        if len(additional_nodes) <= remaining_size:
            selected_additional_nodes = additional_nodes
        else:
            np.random.seed(42)
            selected_additional_nodes = np.random.choice(additional_nodes, size=remaining_size, replace=False)

        selected_nodes = np.concatenate((target_nodes_idx, selected_additional_nodes))
    else:
        selected_nodes = target_nodes_idx

    subgraph_dict = {}
    subgraph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(
        graph_dict['adj_norm'][selected_nodes, :][:, selected_nodes])
    subgraph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(
        graph_dict['adj_label'][selected_nodes, :][:, selected_nodes])
    subgraph_dict['norm_value'] = graph_dict['norm_value']

    sub_expression_tensor = expression_tensor[selected_nodes]
    sub_label_tensor = label_tensor[selected_nodes]

    return subgraph_dict, sub_expression_tensor, sub_label_tensor, selected_nodes


def to_tensor(graph_dict):
    tensor_graph_dict = {}
    tensor_graph_dict['adj_norm'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_norm'])
    tensor_graph_dict['adj_label'] = sparse_mx_to_torch_sparse_tensor(graph_dict['adj_label'])
    tensor_graph_dict['norm_value'] = graph_dict['norm_value']
    return tensor_graph_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


if __name__ == '__main__':
    opt = TrainOptions().parse()

    sc_adata = sc.read(opt.sc_data)
    st_adata = sc.read(opt.st_data)

    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()
    
    target_gene_names = opt.target_genes.strip()
    if target_gene_names not in sc_adata.var_names:
        raise ValueError(f"目标基因 '{target_gene_names}' 不在单细胞数据集中！")
    
    # 保存目标基因在参考数据中的表达值
    target_gene_idx = list(sc_adata.var_names).index(target_gene_names)
    target_gene_expr = sc_adata.X[:, target_gene_idx].copy() if not scipy.sparse.issparse(sc_adata.X) else sc_adata.X[:, target_gene_idx].toarray().copy()
    target_gene_expr = torch.Tensor(target_gene_expr)
    print("len(target_gene_names):", len(target_gene_expr))
    
    # 移除目标基因，使其不参与训练
    sc_adata = sc_adata[:, [g for g in sc_adata.var_names if g != target_gene_names]].copy()

    common_genes = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    if "spatial" in st_adata.obsm:
        spatial_coords = st_adata.obsm["spatial"]  # 空间坐标
    else:
        raise ValueError("空间数据集中没有找到 'spatial' 坐标信息！")

    combined_adata = sc.concat([sc_adata, st_adata], join="inner", label="batch")

    combined_adata.obs["batch"] = ["scRNA" if i < sc_adata.n_obs else "spatial" for i in range(combined_adata.n_obs)]
    combined_adata.obs['original_total_counts'] = np.sum(combined_adata.X, axis=1)

    sc.pp.filter_genes(combined_adata, min_cells=opt.min_cells)  # 过滤低表达基因

    sc.pp.highly_variable_genes(
        combined_adata,
        n_top_genes=opt.highly_variable,
        batch_key="batch",
        flavor="seurat_v3"
    )

    combined_adata = combined_adata[:, combined_adata.var['highly_variable']].copy()

    sc.pp.normalize_total(combined_adata)
    sc.pp.log1p(combined_adata)

    common_sc_adata = combined_adata[combined_adata.obs["batch"] == "scRNA"].copy()
    common_st_adata = combined_adata[combined_adata.obs["batch"] == "spatial"].copy()

    common_st_adata.obsm["spatial"] = spatial_coords[:common_st_adata.n_obs]

    print(f"共有高变基因数量: {len(common_sc_adata.var_names)}")


    print(target_gene_names)

    gene_train = np.array([g for g in common_sc_adata.var_names if g not in target_gene_names])
    print(f"选择 {len(gene_train)} 个基因作为训练基因")


    graph_dict_sc = graph_construction_scRNA(common_sc_adata, n_neighbors=opt.sc_neighbors)

    if scipy.sparse.issparse(common_sc_adata.X):
        x_sc = torch.Tensor(common_sc_adata.X.toarray())
    else:
        x_sc = torch.Tensor(common_sc_adata.X)
    l_sc = torch.Tensor(common_sc_adata.obs.labels)
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
    print(
        '==> Epochs: %d, Batches: %d, sc_batches: %d, st_batches: %d' % (opt.n_epochs, batches, sc_batches, st_batches))

    kfold_randomstate = np.random.RandomState(42)
    kfold = KFold(n_splits=opt.k_folds, shuffle=True, random_state=kfold_randomstate)

    # 存储每个子模型的最佳结果
    best_submodel_results = []
    best_submodel_epochs = []

    for i_submodel, (gene_fit_idx, gene_calibration_idx) in enumerate(kfold.split(gene_train)):
        gene_fit = gene_train[gene_fit_idx]
        gene_calibration = gene_train[gene_calibration_idx]
        gene_calibration_indices = [list(common_sc_adata.var_names).index(gene) for gene in gene_calibration]

        print('==> Training submodel: %d' % (i_submodel + 1))
        print(f"Training genes: {gene_fit}, Calibration genes: {gene_calibration}")
        best = 0
        setup_seed(1)
        model = spaAdaptModel(opt)
        model.setup(opt)

        best_score = 0
        best_imputed = None

        for epoch in range(1, opt.n_epochs + 1):
            sc_cell_idxes = np.arange(sc_cells_number)
            st_cell_idxes = np.arange(st_cells_number)

            train_target_expr = x_st[:, gene_calibration_indices].clone()
            train_x_st = x_st.clone()
            train_x_st[:, gene_calibration_indices] = 0


            for batch in range(batches):
                scb = batch % sc_batches
                stb = batch % st_batches


                scb_cell_idx = sc_cell_idxes[scb * opt.cells_per_batch:(scb + 1) * opt.cells_per_batch]
                stb_cell_idx = st_cell_idxes[stb * opt.cells_per_batch:(stb + 1) * opt.cells_per_batch]

                subgraph_dict_sc, subx_sc, subl_sc, _ = retrieve_fixed_size_subgraph_with_target(graph_dict_sc, x_sc,
                                                                                                 l_sc, scb_cell_idx,
                                                                                                 fixed_size=opt.neighbor_per_batch)
                subgraph_dict_st, subx_st, subl_st, selected_nodes = retrieve_fixed_size_subgraph_with_target(
                    graph_dict_st, train_x_st, l_st, stb_cell_idx, fixed_size=opt.neighbor_per_batch)


                batch_train_expr = train_target_expr[selected_nodes]


                model.optimize_parameters(subx_sc, subl_sc, subgraph_dict_sc, subx_st, subgraph_dict_st,
                                          gene_calibration_indices, batch_train_expr)

                if batch % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    print(
                        '==> epoch: %d, Batch:[%d/%d]  | Loss: cls %.4f, mmd %.4f, GS %.4f, GT %.4f, CA %.4f, DIFF %.4f, Imput%.4f' % (
                            epoch, batch, batches, losses['CLS'], losses['MMD'], losses['GS'], losses['GT'],
                            losses['CA'], losses['DIFF'], losses['Imput']))
            model.update_learning_rate()

            if epoch > 10:
                print('\n===> Start inferencing at epoch %d...' % epoch)
                test_data = {'x': x_st, 'adj_norm': tensor_graph_dict_st['adj_norm']}
                ref_data = {
                    'x': x_sc, 
                    'adj_norm': to_tensor(graph_dict_sc)['adj_norm'],
                }
                imputed_expr, _ = model.impute_genes(test_data, ref_data, target_gene_expr)
                imputed_expr = imputed_expr.cpu().numpy()


                imputed_expr = np.clip(imputed_expr, a_min=None, a_max=100)
                imputed_expr_exp = np.expm1(imputed_expr)
                target_sum = 1e4
                imputed_expr_rescaled = imputed_expr_exp.copy()
                if target_gene_names == 'Mbp' or target_gene_names == 'Pcsk2':
                    for i in range(imputed_expr_exp.shape[0]):
                        scaling_factor = common_st_adata.obs['original_total_counts'].iloc[i] / target_sum
                        imputed_expr_rescaled[i, :] *= scaling_factor

                cell_type_pccs = []
                for cell_type in np.unique(common_st_adata.obs.cell_types):
                    st_type_mask = common_st_adata.obs.cell_types == cell_type
                    sc_type_mask = common_sc_adata.obs.cell_types == cell_type
                    
                    if np.sum(st_type_mask) > 0 and np.sum(sc_type_mask) > 0:
                        st_type_mean = np.mean(imputed_expr_rescaled[st_type_mask])
                        sc_expr = common_sc_adata.X[sc_type_mask]
                        if scipy.sparse.issparse(sc_expr):
                            sc_expr = sc_expr.toarray()
                        if isinstance(target_gene_expr, torch.Tensor):
                            target_gene_expr = target_gene_expr.detach().cpu().numpy()
                        sc_type_mean = np.mean(target_gene_expr)
                        

                        relative_diff = abs(st_type_mean - sc_type_mean) / (sc_type_mean + 1e-10)
                        similarity_score = 1 / (1 + relative_diff)
                        cell_type_pccs.append(similarity_score)

                mean_pcc = np.nanmean(cell_type_pccs)

                if mean_pcc > best_score:
                    best_score = mean_pcc
                    best_imputed = imputed_expr_rescaled
                    print(
                        f'\t### Found best model for submodel {i_submodel + 1} at epoch {epoch} with accuracy: {mean_pcc:.4f}')

        best_submodel_results.append(best_imputed)
        print(f'\n===> Submodel {i_submodel + 1} best accuracy: {best_score:.4f}')

    print('\n===> Aggregating results from all submodels...')
    final_imputed = np.mean(best_submodel_results, axis=0)


    # 保存填补结果
    if scipy.sparse.issparse(st_adata.X):
        original_st_data = st_adata.X.toarray()
    else:
        original_st_data = st_adata.X.copy()

    final_imputed_processed = np.log1p(final_imputed)

    new_var_names = list(st_adata.var_names) + [target_gene_names]
    new_st_data = np.column_stack((original_st_data, final_imputed_processed.flatten()))

    new_var = pd.DataFrame(index=pd.Index(new_var_names))

    new_st_adata = sc.AnnData(
        X=new_st_data,
        obs=st_adata.obs,
        var=new_var,
        uns=st_adata.uns,
        obsm=st_adata.obsm
    )

    output_h5ad_path = f'{opt.name}_imputed_{target_gene_names}.h5ad'
    new_st_adata.write(output_h5ad_path)
    print(f"\n===> 填补结果已保存为: {output_h5ad_path}")


