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
        self.embedding_size = conf.getint('Augmentation', 'embedding_size')
        self.arg_latent_size = conf.getint('Augmentation', 'latent_size')
        self.arg_argmentation_type = eval(conf.get('Augmentation', 'argmentation_type'))
        self.arg_argmentation_num = conf.getint('Augmentation', 'argmentation_num')
        self.is_augmentation = conf.getboolean('Augmentation', 'is_augmentation')
        self.arg_pretrain_lr = conf.getfloat('Augmentation', 'pretrain_lr')
        self.arg_pretrain_epochs = conf.getint('Augmentation', 'pretrain_epochs')
        self.arg_batch_size = conf.getint('Augmentation', 'batch_size')


