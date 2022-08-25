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
from openhgnn import HPN
from rcvae_model import VAE
import torch.nn as nn
import copy

class HPN_AUG(nn.Module):

    def __init__(self, config, g, feature_sizes, category_index, target_category, label_num, dataset, meta_path):
        super().__init__()

        if config.is_augmentation:
            self.model = HPN(meta_path, [target_category], config.embedding_size * (len(config.arg_argmentation_type)) + feature_sizes[category_index[target_category]], label_num, config.dropout, config.k_layer, config.alpha, config.edge_drop)
        else:
            self.model = HPN(meta_path, [target_category], feature_sizes[category_index[target_category]], label_num, config.dropout, config.k_layer, config.alpha, config.edge_drop)
        self.meta_path = meta_path
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
