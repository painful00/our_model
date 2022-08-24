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
from openhgnn import SimpleHGN
from rcvae_model import VAE
import torch.nn as nn
import copy


class SimpleHGN_AUG(nn.Module):

    def __init__(self, config, g, feature_sizes, category_index, target_category, label_num, dataset):
        super().__init__()
        heads = [config.num_heads] * config.num_layers + [1]
        if config.is_augmentation:
            self.model = SimpleHGN(config.edge_dim, len(g.etypes), [config.embedding_size * (len(config.arg_argmentation_type)) + feature_sizes[category_index[target_category]]],
                                   config.hidden_dim, label_num, config.num_layers, heads, config.feat_drop,
                                   config.negative_slope, config.residual, config.beta)
        else:
            self.model = SimpleHGN(config.edge_dim, len(g.etypes), [feature_sizes[category_index[target_category]]], config.hidden_dim, label_num, config.num_layers, heads, config.feat_drop, config.negative_slope, config.residual, config.beta)
        self.look_up_table = []
        self.feature_sizes = feature_sizes
        self.category_index = category_index
        self.target_category = target_category
        self.label_num = label_num
        self.config = config

        if config.is_augmentation:
            for i, size in enumerate(feature_sizes):
                if i == category_index[target_category]:
                    self.look_up_table.append(identical_map)
                else:
                    size_map = config.embedding_size * (len(config.arg_argmentation_type)) + feature_sizes[category_index[target_category]]
                    self.look_up_table.append(nn.Linear(size, size_map))

        else:
            for i, size in enumerate(feature_sizes):
                if i == category_index[target_category]:
                    self.look_up_table.append(identical_map)
                else:
                    self.look_up_table.append(nn.Linear(size, feature_sizes[category_index[target_category]]))
            if dataset == "yelp":
                self.look_up_table = [identical_map for _ in range(len(feature_sizes))]

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
            logits = self.model(g, feat_dict)[self.target_category]

        else:
            for ntype in g_ori.ntypes:
                feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g_ori.ndata["h"][ntype])
            logits = self.model(g_ori, feat_dict)[self.target_category]
        return logits

def identical_map(x):
    return x


class SimpleHGN_AUG_P(nn.Module):

    def __init__(self, config, g1, g2, feature_sizes, category_index, target_category, label_num, dataset):
        super().__init__()
        heads = [config.num_heads] * config.num_layers + [1]

        if config.is_augmentation:
            self.model = SimpleHGN(config.edge_dim, len(g1.etypes)+len(g2.etypes), [feature_sizes[category_index[target_category]]], config.hidden_dim, label_num, config.num_layers, heads, config.feat_drop, config.negative_slope, config.residual, config.beta)
        else:
            self.model = SimpleHGN(config.edge_dim, len(g2.etypes), [feature_sizes[category_index[target_category]]],
                                   config.hidden_dim, label_num, config.num_layers, heads, config.feat_drop,
                                   config.negative_slope, config.residual, config.beta)


        self.look_up_table = []
        self.feature_sizes = feature_sizes
        self.category_index = category_index
        self.target_category = target_category
        self.label_num = label_num
        self.config = config

        self.mapping = nn.Linear(2*label_num, label_num)

        for i, size in enumerate(feature_sizes):
            if i == category_index[target_category]:
                self.look_up_table.append(identical_map)
            else:
                self.look_up_table.append(nn.Linear(size, feature_sizes[category_index[target_category]]))
        if dataset == "yelp":
            self.look_up_table = [identical_map for _ in range(len(feature_sizes))]


    def forward(self, g_aug, g_ori):
        feat_dict = {}
        if self.config.is_augmentation:
            for ntype in g_aug.ntypes:
                if ntype in self.category_index:
                    feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g_aug.ndata["h"][ntype])
                else:
                    feat_dict[ntype] = g_aug.ndata["h"][ntype]
            logits1 = self.model(g_aug, feat_dict)[self.target_category]

            feat_dict = {}
            for ntype in g_ori.ntypes:
                if ntype in self.category_index:
                    feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g_ori.ndata["h"][ntype])
            logits2 = self.model(g_ori, feat_dict)[self.target_category]

            logits = F.softmax(self.mapping(torch.cat((logits1,logits2), dim=-1)), dim=-1)


        else:
            for ntype in g_ori.ntypes:
                feat_dict[ntype] = self.look_up_table[self.category_index[ntype]](g_ori.ndata["h"][ntype])
            logits = self.model(g_ori, feat_dict)[self.target_category]

        return logits