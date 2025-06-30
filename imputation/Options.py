import argparse
import os
import torch
import utils
import time

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--name', type=str, default='HPR', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--input_dim', type=int, default=153, help='Dim of input, using HVG')
        parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of encoder hidden layer 1.')
        parser.add_argument('--feat_hidden2', type=int, default=40, help='Dim of encoder hidden layer 2.')
        parser.add_argument('--gcn_hidden1', type=int, default=40, help='Dim of VGAE hidden layer 1.')
        parser.add_argument('--gcn_hidden2', type=int, default=20, help='Dim of VGAE hidden layer 2.')
        parser.add_argument('--share_hidden', type=int, default=30, help='Dim of share and private hidden layer.')
        parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')

        parser.add_argument('--sc_neighbors', type=int, default=20, help='K nearest neighbors to create sc graph.')
        parser.add_argument('--st_neighbors', type=int, default=20, help='K nearest neighbors to create st graph.')
        parser.add_argument('--dis_sigma', type=int, default=45, help='dis sigma of  nearest neighbors to weight st graph.')
        parser.add_argument('--distance_type', type=str, default='euclidean', help='graph distance type: [euclidean | cosine | correlation].')
        parser.add_argument('--min_cells', type=int, default=3,help='filter genes.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--load_epoch', type=str, default=40, help='which epoch to load? set to latest to use latest cached model')


        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Training data parameters
        parser.add_argument('--sc_data', type=str, default='/home/lsy/data/merfish/Moffit_RNA.h5ad',help='source scRNA-seq dataset, h5ad format.')
        parser.add_argument('--st_data', type=str, default='/home/lsy/data/merfish/MERFISH_1.h5ad',help='target spatial transcriptomic dataset, h5ad format.')

        # Discriminator parameters
        parser.add_argument('--neighbor_per_batch', type=int, default=300, help='max neighbor nodes in per batch')
        parser.add_argument('--highly_variable', type=int, default=1000, help='# highly variable gene.')
        parser.add_argument('--target_genes', type=str, default='Gad2', help='target genes to impute.')
        parser.add_argument('--topK', type=int, default=50, help='topK')
        parser.add_argument('--k_folds', type=int, default=5, help='k_folds')

        # Training parameters
        parser.add_argument('--phase', type=str, default='train', help='train')
        parser.add_argument('--classes', type=int, default=9, help='number of classes for CLS branch')
        parser.add_argument('--alpha', type=str, default=None,help='list of alpha values for Focal Loss')
        parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gamma', type=float, default=0.5, help='step lr decay gamma')
        parser.add_argument('--lr_decay_epochs', type=int, default=10, help='multiply by a gamma every lr_decay_epochs epochs')
        parser.add_argument('--cells_per_batch', type=int, default=150, help='random sampling #number of cells per epoch')

        parser.add_argument('--lambda_F', type=float, default=20, help='Weight of reconstruction loss.')
        parser.add_argument('--lambda_CA', type=float, default=0.2, help='Weight for class consistency loss.')
        parser.add_argument('--lambda_C', type=float, default=1, help='Weight of classification loss.')
        parser.add_argument('--lambda_Dep', type=float, default=0.1, help='Weight of disentanglement loss.')


        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=40, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
        parser.add_argument('--print_freq', type=int, default=100, help='batch frequency of printing the loss')

        self.isTrain = True
        return parser
