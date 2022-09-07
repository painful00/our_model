import configparser
import numpy as np
import torch

class Config(object):
    def __init__(self, file_path, model, dataset, gpu, augmenter):
        conf = configparser.ConfigParser()
        if gpu == -1:
            self.device = torch.device('cpu')
        elif gpu >= 0:
            if torch.cuda.is_available():
                self.device = torch.device('cuda', int(gpu))
            else:
                raise ValueError("cuda is not available, please set 'gpu' -1")

        try:
            conf.read(file_path)
        except:
            print("failed!")

        # training dataset path
        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.model = model
        self.dataset = dataset
        self.optimizer = 'Adam'
        # model
        if self.model == 'HAN':
            self.lr = conf.getfloat("HAN", "learning_rate")
            self.weight_decay = conf.getfloat("HAN", "weight_decay")
            self.seed = conf.getint("HAN", "seed")
            self.dropout = conf.getfloat("HAN", "dropout")
            self.hidden_dim = conf.getint('HAN', 'hidden_dim')
            self.out_dim = conf.getint('HAN', 'out_dim')
            num_heads = conf.get('HAN', 'num_heads').split('-')
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint('HAN', 'patience')
            self.max_epoch = conf.getint('HAN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")

        elif self.model == 'MAGNN':
            self.lr = conf.getfloat("MAGNN", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN", "weight_decay")
            self.seed = conf.getint("MAGNN", "seed")
            self.dropout = conf.getfloat("MAGNN", "dropout")
            self.hidden_dim = conf.getint('MAGNN', 'h_dim')
            self.out_dim = conf.getint('MAGNN', 'out_dim')
            self.inter_attn_feats = conf.getint('MAGNN', 'inter_attn_feats')
            self.num_heads = conf.getint('MAGNN', 'num_heads')
            self.patience = conf.getint('MAGNN', 'patience')
            self.max_epoch = conf.getint('MAGNN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")
            self.num_layers = conf.getint('MAGNN', 'num_layers')
            self.batch_size = conf.getint('MAGNN', 'batch_size')
            self.num_samples = conf.getint('MAGNN', 'num_samples')
            self.encoder_type = conf.get('MAGNN', 'encoder_type')

        elif self.model == 'SimpleHGN':
            self.lr = conf.getfloat("SimpleHGN", "lr")
            self.weight_decay = conf.getfloat("SimpleHGN", "weight_decay")
            self.seed = conf.getint("SimpleHGN", "seed")
            self.feat_drop = conf.getfloat("SimpleHGN", "feats_drop_rate")
            self.hidden_dim = conf.getint('SimpleHGN', 'h_dim')
            self.negative_slope = conf.getfloat('SimpleHGN', 'slope')
            self.edge_dim = conf.getint('SimpleHGN', 'edge_dim')
            self.max_epoch = conf.getint('SimpleHGN', 'max_epoch')
            self.patience = conf.getint('SimpleHGN', 'patience')
            self.beta = conf.getfloat("SimpleHGN", "beta")
            self.num_layers = conf.getint('SimpleHGN', 'n_layers')
            self.residual = conf.getboolean('SimpleHGN', 'residual')
            self.num_heads = conf.getint('SimpleHGN', 'num_heads')

        elif self.model == "HGT":
            self.lr = conf.getfloat("HGT", "learning_rate")
            self.weight_decay = conf.getfloat("HGT", "weight_decay")
            self.seed = conf.getint("HGT", "seed")
            self.dropout = conf.getfloat("HGT", "dropout")
            self.batch_size = conf.getint('HGT', 'batch_size')
            self.patience = conf.getint('HGT', 'patience')
            self.hidden_dim = conf.getint('HGT', 'hidden_dim')
            self.out_dim = conf.getint('HGT', 'out_dim')
            self.num_layers = conf.getint('HGT', 'num_layers')
            self.num_heads = conf.getint('HGT', 'num_heads')
            self.num_workers = conf.getint('HGT', 'num_workers')
            self.max_epoch = conf.getint('HGT', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")
            self.norm = conf.getboolean("HGT", "norm")

        elif self.model == "HPN":
            self.lr = conf.getfloat("HPN", "learning_rate")
            self.weight_decay = conf.getfloat("HPN", "weight_decay")
            self.seed = conf.getint("HPN", "seed")
            self.dropout = conf.getfloat("HPN", "dropout")
            self.k_layer = conf.getint('HPN', 'k_layer')
            self.patience = conf.getint('HPN', 'patience')
            self.hidden_dim = conf.getint('HPN', 'hidden_dim')
            self.out_dim = conf.getint('HPN', 'out_dim')
            self.alpha = conf.getfloat('HPN', 'alpha')
            self.edge_drop = conf.getfloat('HPN', 'edge_drop')
            self.max_epoch = conf.getint('HPN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")

        elif self.model == "CompGCN":
            self.lr = conf.getfloat("CompGCN", "learning_rate")
            self.weight_decay = conf.getfloat("CompGCN", "weight_decay")
            self.seed = conf.getint("CompGCN", "seed")
            self.dropout = conf.getfloat("CompGCN", "dropout")
            self.num_layers = conf.getint('CompGCN', 'n_layers')
            self.patience = conf.getint('CompGCN', 'patience')
            self.in_dim = conf.getint('CompGCN', 'in_dim')
            self.hidden_dim = conf.getint('CompGCN', 'hidden_dim')
            self.out_dim = conf.getint('CompGCN', 'out_dim')
            self.comp_fn = conf.get('CompGCN', 'comp_fn')
            self.max_epoch = conf.getint('CompGCN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("CompGCN", "mini_batch_flag")
            self.validation = conf.getboolean("CompGCN", "validation")


        # augmentation generator
        self.is_augmentation = conf.getboolean('Augmentation', 'is_augmentation')
        self.arg_argmentation_type = []
        self.arg_argmentation_num = 0
        if augmenter == "STR_META":
            # structure
            self.embedding_size = conf.getint('Augmentation', 'embedding_size')
            self.arg_latent_size = conf.getint('Augmentation', 'latent_size')
            self.arg_argmentation_type = eval(conf.get('Augmentation', 'argmentation_type'))
            self.arg_argmentation_num = conf.getint('Augmentation', 'argmentation_num')
            self.arg_pretrain_lr = conf.getfloat('Augmentation', 'pretrain_lr')
            self.arg_pretrain_epochs = conf.getint('Augmentation', 'pretrain_epochs')
            self.arg_batch_size = conf.getint('Augmentation', 'batch_size')

            # path
            self.resolution = conf.getint('Augmentation', 'resolution')
            self.threshold_sba = conf.getfloat('Augmentation', 'threshold_sba')
            self.threshold_usvt = conf.getfloat('Augmentation', 'threshold_usvt')
            self.alpha = conf.getfloat('Augmentation', 'alpha')
            self.beta = conf.getfloat('Augmentation', 'beta')
            self.gamma = conf.getfloat('Augmentation', 'gamma')
            self.inner_iters = conf.getint('Augmentation', 'inner_iters')
            self.outer_iters = conf.getint('Augmentation', 'outer_iters')
            self.n_trials = conf.getint('Augmentation', 'n_trials')
            self.argmentation_path = eval(conf.get('Augmentation', 'argmentation_path'))
            self.argmentation_intra_graph_num = conf.getint('Augmentation', 'argmentation_intra_graph_num')
            self.argmentation_inter_graph_num = conf.getint('Augmentation', 'argmentation_inter_graph_num')

        elif augmenter == "dropedge":
            self.dropedge_rate = conf.getfloat('DropEdge', 'dropedge_rate')

        elif augmenter == "LA":
            self.arg_batch_size = conf.getint('LA', 'batch_size')
            self.embedding_size = conf.getint('LA', 'embedding_size')
            self.arg_latent_size = conf.getint('LA', 'latent_size')
            self.arg_pretrain_lr = conf.getfloat('LA', 'pretrain_lr')
            self.arg_pretrain_epochs = conf.getint('LA', 'pretrain_epochs')
            self.arg_argmentation_num = conf.getint('LA', 'arg_argmentation_num')

