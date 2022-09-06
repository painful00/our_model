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
        self.config = config

        # self.model = MAGNN(ntypes=ntypes,
        #                    h_feats=config.hidden_dim,
        #                    inter_attn_feats=config.inter_attn_feats,
        #                    num_heads=config.num_heads,
        #                    num_classes=label_num,
        #                    num_layers=config.num_layers,
        #                    metapath_list=metapath_list,
        #                    edge_type_list=edge_type_list,
        #                    dropout_rate=config.dropout,
        #                    encoder_type=config.encoder_type,
        #                    metapath_idx_dict=metapath_idx_dict)
        if config.is_augmentation:
            self.model = MAGNN(ntypes=ntypes,
                               h_feats=feature_sizes[category_index[target_category]]+config.embedding_size * (len(config.arg_argmentation_type)),
                               inter_attn_feats=config.inter_attn_feats,
                               num_heads=1,
                               num_classes=label_num,
                               num_layers=config.num_layers,
                               metapath_list=metapath_list,
                               edge_type_list=edge_type_list,
                               dropout_rate=config.dropout,
                               encoder_type=config.encoder_type,
                               metapath_idx_dict=metapath_idx_dict)
        else:
            self.model = MAGNN(ntypes=ntypes,
                               h_feats=feature_sizes[category_index[target_category]],
                               inter_attn_feats=config.inter_attn_feats,
                               num_heads=1,
                               num_classes=label_num,
                               num_layers=config.num_layers,
                               metapath_list=metapath_list,
                               edge_type_list=edge_type_list,
                               dropout_rate=config.dropout,
                               encoder_type=config.encoder_type,
                               metapath_idx_dict=metapath_idx_dict)

        if config.is_augmentation:
            for i, size in enumerate(feature_sizes):
                if i == category_index[target_category]:
                    size = config.embedding_size * (len(config.arg_argmentation_type)) + feature_sizes[
                        category_index[target_category]]
                self.look_up_table.append(nn.Linear(size, config.num_heads*config.hidden_dim))
                # if i == category_index[target_category]:
                #     self.look_up_table.append(identical_map)
                # else:
                #     size_map = config.embedding_size * (len(config.arg_argmentation_type)) + feature_sizes[category_index[target_category]]
                #     self.look_up_table.append(nn.Linear(size, size_map))

        else:
            for i, size in enumerate(feature_sizes):
                if i == category_index[target_category]:
                    self.look_up_table.append(identical_map)
                else:
                    self.look_up_table.append(nn.Linear(size, feature_sizes[category_index[target_category]]))
                # if i == category_index[target_category]:
                #     size = feature_sizes[category_index[target_category]]
                # self.look_up_table.append(nn.Linear(size, config.num_heads*config.hidden_dim))


    def forward(self, g_ori, augmentated_features, augmentated_types, augmentated_num, method):
        feat_dict = {}
        if self.config.is_augmentation:
            dealed_augmentated_features = {}
            for aug_type in augmentated_types:
                temp_features = None
                for i in range(augmentated_num):
                    if method == "mean":
                        if temp_features is not None:
                            temp_features = torch.add(temp_features, augmentated_features[aug_type][i])
                        else:
                            temp_features = augmentated_features[aug_type][i]
                    elif method == "concate":
                        if temp_features is not None:
                            temp_features = torch.cat((temp_features, augmentated_features[aug_type][i]), dim=-1)
                        else:
                            temp_features = augmentated_features[aug_type][i]
                if method == "mean":
                    temp_features = temp_features / augmentated_num

                dealed_augmentated_features[aug_type] = temp_features

            # deep copy
            g = copy.deepcopy(g_ori)

            # concate
            for aug_type in augmentated_types:
                g.nodes[self.target_category].data["h"] = torch.cat(
                    (g.ndata["h"][self.target_category], dealed_augmentated_features[aug_type]), dim=-1)
            for ntype in g_ori.ntypes:
                feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g.ndata["h"][ntype])
            logits = self.model(g_ori, feat_dict)[self.target_category]

        else:
            for ntype in g_ori.ntypes:
                feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g_ori.ndata["h"][ntype])
            logits = self.model(g_ori, feat_dict)[self.target_category]
        return logits




class MAGNN_AUG_P(nn.Module):

    def __init__(self, config, label_num, g, dataset, target_category, feature_sizes, category_index):
        super().__init__()
        if dataset == "imdb":
            metapath_list = ['M-D-M', 'M-A-M']
            edge_type_list = ["M-D", "D-M", "M-A", "A-M"]
            if config.is_augmentation:
                for ind in range(len(g.ntypes)-3):
                    metapath_list.append('M-M'+str(ind)+"-M")
                    edge_type_list.append('M-M'+str(ind))
                    edge_type_list.append('M'+str(ind)+'-M')
        elif dataset == "acm":
            metapath_list = ["paper-author-paper", "paper-subject-paper"]
            edge_type_list = ["paper-author", "author-paper", "paper-subject", "subject-paper"]
            if config.is_augmentation:
                for ind in range(len(g.ntypes)-3):
                    metapath_list.append('paper-paper'+str(ind)+"-paper")
                    edge_type_list.append('paper-paper'+str(ind))
                    edge_type_list.append('paper'+str(ind)+'-paper')
        elif dataset == "yelp":
            metapath_list = ["b-s-b", "b-u-b", "b-u-b-l-b", "b-u-b-s-b"]
            edge_type_list = ["b-l", "l-b", "b-s", "s-b", "b-u", "u-b"]
            if config.is_augmentation:
                for ind in range(len(g.ntypes)-4):
                    metapath_list.append('b-b'+str(ind)+"-b")
                    edge_type_list.append('b-b'+str(ind))
                    edge_type_list.append('b'+str(ind)+'-b')

        metapath_idx_dict = mp_instance_sampler(g, metapath_list, dataset)
        ntypes = g.ntypes
        self.target_category = target_category
        self.look_up_table = []
        self.feature_sizes = feature_sizes
        self.category_index = category_index
        self.config = config

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


    def forward(self, new_g):

        feat_dict = {}
        for ntype in new_g.ntypes:
            feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](new_g.ndata["h"][ntype])
        logits = self.model(new_g, feat_dict)[self.target_category]

        return logits

def identical_map(x):
    return x
