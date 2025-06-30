import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from module import VGAE, Classifier, EuclideanAttention
from loss import FocalLoss, MixRBFMMDLoss, DiffLoss, ClassAlignmentLoss, ImputLoss
from utils import get_scheduler
from collections import OrderedDict
import numpy as np


class spaAdaptModel(nn.Module):

    def __init__(self, opt):
        super(spaAdaptModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir


        self.loss_names = ['MMD', 'GS', 'GT', 'CLS', 'CA','DIFF', 'Imput']
        self.optimizers = []

        self.netG = VGAE(opt).to(self.device)
        self.netC = Classifier(opt).to(self.device)
        self.model_names = ['G', 'C']
        self.attn = EuclideanAttention(topk=opt.topK, method='softmax')

        if self.isTrain:  # define discriminators
            self.criterionREC_F = torch.nn.MSELoss()
            self.criterionCLS = FocalLoss()
            self.class_alignment_loss = ClassAlignmentLoss()
            self.DiffLoss = DiffLoss()
            base = 1.0  # sigma for MMD
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list]
            self.MMDLoss = MixRBFMMDLoss(sigma_list=sigma_list)
            self.imputloss = ImputLoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_C)

            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]



    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def backward_MMD(self, x_s, l_s, graph_dict_s, x_t, graph_dict_t):
        _, _, share_z_s, _, _ = self.netG(x_s.to(self.device), graph_dict_s['adj_norm'].to(self.device),is_source=True)
        _, _, share_z_t, _, _ = self.netG(x_t.to(self.device), graph_dict_t['adj_norm'].to(self.device),is_source=False)

        # 计算MMD损失
        self.loss_MMD = self.MMDLoss(share_z_s, share_z_t)

        self.loss_CA = self.class_alignment_loss(share_z_s, share_z_t)
        # 计算总损失
        total_loss = self.loss_MMD + self.opt.lambda_CA * self.loss_CA
        total_loss.backward()

    def backward_G_S(self, x_s, l_s, graph_dict_s):
        lambda_F = self.opt.lambda_F
        lambda_C = self.opt.lambda_C
        lambda_Dep = self.opt.lambda_Dep

        gcn_mu_s, gcn_logstd_s, share_z_s, private_z_s, decoded_x_s = self.netG(x_s.to(self.device),
                                                                                graph_dict_s['adj_norm'].to(
                                                                                    self.device),is_source=True)

        loss_rec_s = self.criterionREC_F(decoded_x_s, x_s.to(self.device))
        pred1 = self.netC(share_z_s)

        loss_cls_s = self.criterionCLS(pred1, l_s.long().to(self.device))
        loss_diff = self.DiffLoss(share_z_s, private_z_s)

        self.loss_GS = lambda_F * loss_rec_s
        self.loss_CLS = lambda_C * loss_cls_s
        self.loss_DIFF = lambda_Dep * loss_diff

        self.loss_G = self.loss_GS + self.loss_CLS + self.loss_DIFF
        self.loss_G.backward()

    def backward_G_T(self, x_t, graph_dict_t):
        lambda_F = self.opt.lambda_F
        lambda_Dep = self.opt.lambda_Dep
        gcn_mu_t, gcn_logstd_t, share_z_t, private_z_t, decoded_x_t = self.netG(x_t.to(self.device),
                                                                                graph_dict_t['adj_norm'].to(
                                                                                    self.device),is_source=False)
        loss_rec_t = self.criterionREC_F(decoded_x_t, x_t.to(self.device))
        loss_diff = self.DiffLoss(share_z_t, private_z_t)

        self.loss_GT = lambda_F * loss_rec_t + lambda_Dep * loss_diff
        self.loss_GT.backward()

    def backward_impute(self, x_s, graph_dict_s, x_t, graph_dict_t, target_genes, query_supervision):

        self.netG.train()
        self.set_requires_grad(self.netC, False)
        self.set_requires_grad(self.netG,True)
        self.optimizer_G.zero_grad()
        _, _, query_share_z, _, _ = self.netG(x_t.to(self.device),
                                              graph_dict_t['adj_norm'].to(self.device),is_source=False)

        _, _, ref_share_z, _, _ = self.netG(x_s.to(self.device),
                                            graph_dict_s['adj_norm'].to(self.device),is_source=True)

        ref_gene_expr = x_s[:, target_genes].to(self.device)

        imputed_expr, attention_weights = self.netG.impute_genes(query_share_z, ref_share_z, ref_gene_expr)
        query_supervision = query_supervision.to(self.device)

        self.loss_Imput = self.imputloss(imputed_expr, query_supervision)
        self.loss_Imput.backward()
        self.optimizer_G.step()

    def optimize_parameters(self, x_s, l_s, graph_dict_s, x_t, graph_dict_t,target_genes, query_supervision):
        self.netG.train()
        self.netC.train()

        # Step 1. Training G and C with source domain.
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netC, True)
        self.optimizer_G.zero_grad()
        self.optimizer_C.zero_grad()
        self.backward_G_S(x_s, l_s, graph_dict_s)
        self.optimizer_G.step()
        self.optimizer_C.step()

        # Step 2. Training G with target domain.
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netC, False)
        self.optimizer_G.zero_grad()
        self.backward_G_T(x_t, graph_dict_t)
        self.optimizer_G.step()

        # Step 3. Training G with two domains.
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netC, False)
        self.optimizer_G.zero_grad()
        self.backward_MMD(x_s, l_s, graph_dict_s, x_t, graph_dict_t)
        self.optimizer_G.step()

        #step 4. Train G impute
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netC, False)
        self.optimizer_G.zero_grad()
        self.backward_impute(x_s, graph_dict_s, x_t, graph_dict_t,target_genes,query_supervision)
        self.optimizer_G.step()


    def impute_genes(self, query_data, ref_data, target_gene_expr):
        self.netG.eval()
        with torch.no_grad():
            _, _, query_share_z, _, _ = self.netG(query_data['x'].to(self.device),
                                                 query_data['adj_norm'].to(self.device),is_source=False)

            _, _, ref_share_z, _, _ = self.netG(ref_data['x'].to(self.device),
                                               ref_data['adj_norm'].to(self.device),is_source=True)

            if isinstance(target_gene_expr, np.ndarray):
                ref_gene_expr = torch.from_numpy(target_gene_expr).float().to(self.device)
            else:
                ref_gene_expr = target_gene_expr.to(self.device)

            imputed_expr, attention_weights = self.netG.impute_genes(query_share_z, ref_share_z, ref_gene_expr)
            
        return imputed_expr, attention_weights

    def setup(self, opt):
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_epoch)
        self.print_networks(opt.verbose)


    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret