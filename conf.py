import configparser
import numpy as np
import torch

class Config(object):
    def __init__(self, file_path, model, dataset, gpu):
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

        # augmentation generator
        # structure
        self.embedding_size = conf.getint('Augmentation', 'embedding_size')
        self.arg_latent_size = conf.getint('Augmentation', 'latent_size')
        self.arg_argmentation_type = eval(conf.get('Augmentation', 'argmentation_type'))
        self.arg_argmentation_num = conf.getint('Augmentation', 'argmentation_num')
        self.is_augmentation = conf.getboolean('Augmentation', 'is_augmentation')
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





