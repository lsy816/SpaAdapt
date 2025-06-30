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
        #input_dim = opt.feat_hidden2 + opt.gcn_hidden2
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
        self.p_s_encoder.add_module('p_s_encoder', full_block(opt.gcn_hidden2 + opt.feat_hidden2, self.latent_dim - opt.share_hidden, opt.p_drop))

        self.p_t_encoder = nn.Sequential()
        self.p_t_encoder.add_module('p_t_encoder', full_block(opt.gcn_hidden2 + opt.feat_hidden2,
                                                                      self.latent_dim - opt.share_hidden, opt.p_drop))



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