import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.view(-1, 1)).squeeze()
        pt = pt.gather(1, target.view(-1, 1)).squeeze()

        # 如果提供了 alpha，则应用类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, list):
                alpha = torch.tensor(self.alpha, device=input.device)[target]
            else:
                alpha = self.alpha
            logpt = alpha * logpt

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

class MixRBFMMDLoss(nn.Module):
    def __init__(self, sigma_list, biased=True):
        super().__init__()
        self.sigma_list = sigma_list
        self.biased = biased

    def forward(self, X, Y):
        return mix_rbf_mmd2(X, Y, self.sigma_list, biased=self.biased)


class ClassAlignmentLoss(nn.Module):
    def __init__(self):
        super(ClassAlignmentLoss, self).__init__()

    def _compute_distance_matrix(self, features):

        n = features.size(0)
        dist_matrix = torch.cdist(features, features, p=2)
        dist_matrix = dist_matrix / dist_matrix.max()
        return dist_matrix

    def forward(self, source_features, target_features):
        source_features = F.normalize(source_features, dim=1)
        target_features = F.normalize(target_features, dim=1)

        source_dist = self._compute_distance_matrix(source_features)
        target_dist = self._compute_distance_matrix(target_features)

        structure_loss = F.mse_loss(source_dist, target_dist)

        total_loss =  structure_loss

        return total_loss

class DiffLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(DiffLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def _center_features(self, x):
        return x - x.mean(dim=0, keepdim=True)

    def _compute_adaptive_covariance(self, x, y):
        N = x.size(0)
        x_centered = self._center_features(x)
        y_centered = self._center_features(y)

        x_dim = x_centered.size(1)
        y_dim = y_centered.size(1)

        x_u, _, _ = torch.svd(x_centered)
        y_u, _, _ = torch.svd(y_centered)

        min_dim = min(x_dim, y_dim)
        x_u = x_u[:, :min_dim]
        y_u = y_u[:, :min_dim]

        cov = torch.mm(x_u.t(), y_u) / (N - 1)
        return cov

    def _covariance_loss(self, cov):
        return torch.norm(cov, p='fro') ** 2 / cov.size(0)

    def _feature_similarity_loss(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        sim_matrix = torch.mm(x_norm, y_norm.t())
        sim_diag = torch.diagonal(sim_matrix)

        pos_sim = sim_diag.mean()
        neg_sim = (sim_matrix.sum() - sim_diag.sum()) / (sim_matrix.numel() - sim_matrix.size(0))
        
        return neg_sim - pos_sim

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        cov = self._compute_adaptive_covariance(input1, input2)

        orthogonal_loss = self._covariance_loss(cov)

        diff_loss = self._feature_similarity_loss(input1, input2)

        dim_ratio = min(input1.size(1), input2.size(1)) / max(input1.size(1), input2.size(1))
        adaptive_alpha = self.alpha * dim_ratio

        total_loss = adaptive_alpha * orthogonal_loss + self.beta * diff_loss

        return total_loss


class ImputLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        x_std = torch.std(x, dim=0)
        y_std = torch.std(y, dim=0)
        r = torch.mean((x - x_mean) * (y - y_mean), dim=0) / (x_std * y_std + 1e-6)
        r = torch.mean(r)
        loss = 1 - r
        return loss