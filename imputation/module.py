import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()

        input_dim = opt.share_hidden
        hidden_dim = 30
        dropout = 0.5
        output_dim = opt.classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class DistanceScore(nn.Module):

    def __init__(self, topk=None, method='softmax'):
        super().__init__()
        self.topk = topk
        assert method in ['softmax', 'linear']
        self.method = method

    def forward(self, X1, X2):
        neg_dist_matrix = -torch.cdist(X1, X2, p=2)
        out = torch.zeros_like(neg_dist_matrix)

        if self.method == 'softmax':
            if self.topk is None:
                out = F.softmax(neg_dist_matrix, dim=1)
            else:
                val, idx = torch.topk(neg_dist_matrix, k=self.topk, dim=1)
                score = F.softmax(val, dim=1)
                out = out.scatter_(dim=1, index=idx, src=score)

        elif self.method == 'linear':
            if self.topk is None:
                out = (1 - neg_dist_matrix / neg_dist_matrix.sum(axis=1, keepdim=True)) / (neg_dist_matrix.shape[1] - 1)
            else:
                val, idx = torch.topk(neg_dist_matrix, k=self.topk, dim=1)
                score = (1 - neg_dist_matrix / neg_dist_matrix.sum(axis=1, keepdim=True)) / (
                            neg_dist_matrix.shape[1] - 1)
                out = out.scatter_(dim=1, index=idx, src=score)

        return out, score


class EuclideanAttention(nn.Module):

    def __init__(self, topk=None, method='softmax'):
        super().__init__()
        self.attn_score = DistanceScore(topk=topk, method=method)

    def forward(self, query, key, value):
        attn_weights, score = self.attn_score(query, key)
        out = torch.matmul(attn_weights, value)
        return out, score

# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, training, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.training = training
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output




class VGAE(nn.Module):
    def __init__(self, opt):
        super(VGAE, self).__init__()
        self.training = opt.isTrain
        self.latent_dim = opt.feat_hidden2 + opt.gcn_hidden2

        # Feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(opt.input_dim, opt.feat_hidden1, opt.p_drop))
        self.encoder.add_module('encoder_L2', full_block(opt.feat_hidden1, opt.feat_hidden2, opt.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L1', full_block(self.latent_dim, opt.input_dim, opt.p_drop))

        # GCN layers
        self.gc1 = GraphConvolution(opt.feat_hidden2, opt.gcn_hidden1, opt.isTrain, opt.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(opt.gcn_hidden1, opt.gcn_hidden2, opt.isTrain, opt.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(opt.gcn_hidden1, opt.gcn_hidden2, opt.isTrain, opt.p_drop, act=lambda x: x)

        self.share_encoder = nn.Sequential()
        self.share_encoder.add_module('share_encoder', full_block(opt.gcn_hidden2 + opt.feat_hidden2, opt.share_hidden, opt.p_drop))

        self.p_s_encoder = nn.Sequential()
        self.p_s_encoder.add_module('p_s_encoder',
                                    full_block(opt.gcn_hidden2 + opt.feat_hidden2, self.latent_dim - opt.share_hidden,
                                               opt.p_drop))

        self.p_t_encoder = nn.Sequential()
        self.p_t_encoder.add_module('p_t_encoder', full_block(opt.gcn_hidden2 + opt.feat_hidden2,
                                                              self.latent_dim - opt.share_hidden, opt.p_drop))

        # Gene imputation attention
        self.gene_attention = EuclideanAttention(topk=opt.topK, method='softmax')



    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,is_source=True):
        gcn_mu, gcn_logstd, encoded_x = self.encode(x, adj)
        gcn_z = self.reparameterize(gcn_mu, gcn_logstd)
        z = torch.cat((encoded_x, gcn_z), 1)
        z_share = self.share_encoder(z)
        if is_source:
            z_private = self.p_s_encoder(z)
        else:
            z_private = self.p_t_encoder(z)
        combined_features = torch.cat((z_share, z_private), dim=1)
        decoded_x = self.decoder(combined_features)
        return gcn_mu, gcn_logstd, z_share, z_private, decoded_x

    def impute_genes(self, query_features, ref_features, ref_gene_expr):
        imputed_expr, attention_weights = self.gene_attention(query_features, ref_features, ref_gene_expr)
        return imputed_expr, attention_weights