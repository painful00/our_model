import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import rcvae_pretrain
from utils import load_data, feature_tensor_normalize
import os
from tqdm import trange
from conf import Config
import dgl
from utils import load_data, score, EarlyStopping
from openhgnn import MAGNN
from rcvae_model import VAE
import torch.nn as nn
import copy
from openhgnn.models.MAGNN import mp_instance_sampler

class MAGNN_AUG(nn.Module):

    def __init__(self, config, label_num, g, dataset, target_category, feature_sizes, category_index):
        super().__init__()
        if dataset == "imdb":
            metapath_list = ['M-D-M', 'M-A-M']
            edge_type_list = ["M-D", "D-M", "M-A", "A-M"]
        elif dataset == "acm":
            metapath_list = ["paper-author-paper", "paper-subject-paper"]
            edge_type_list = ["paper-author", "author-paper", "paper-subject", "subject-paper"]
        elif dataset == "yelp":
            metapath_list = ["b-s-b", "b-u-b", "b-u-b-l-b", "b-u-b-s-b"]
            edge_type_list = ["b-l", "l-b", "b-s", "s-b", "b-u", "u-b"]

        metapath_idx_dict = mp_instance_sampler(g, metapath_list, dataset)
        ntypes = g.ntypes
        self.target_category = target_category
        self.look_up_table = []
        self.feature_sizes = feature_sizes
        self.category_index = category_index

        self.model = MAGNN(ntypes=ntypes,
                           h_feats=config.hidden_dim,
                           inter_attn_feats=config.inter_attn_feats,
                           num_heads=config.num_heads,
                           num_classes=label_num,
                           num_layers=config.num_layers,
                           metapath_list=metapath_list,
                           edge_type_list=edge_type_list,
                           dropout_rate=config.dropout,
                           encoder_type=config.encoder_type,
                           metapath_idx_dict=metapath_idx_dict)

        for i, size in enumerate(feature_sizes):
            self.look_up_table.append(nn.Linear(size, config.num_heads*config.hidden_dim))


    def forward(self, g):
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g.ndata["h"][ntype])
        logits = self.model(g, feat_dict)[self.target_category]
        return logits
