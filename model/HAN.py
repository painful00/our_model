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
from openhgnn import HAN
from rcvae_model import VAE
import torch.nn as nn
import copy


class HAN_AUG(nn.Module):

    def __init__(self, config, meta_paths, target_category, hidden_dim, label_num, num_heads, dropout, feature_sizes, mapping_size, category_index, augmentated_types, arg_argmentation_num):
        super().__init__()

        #self.model = HAN(meta_paths, [target_category], mapping_size * (len(augmentated_types))+feature_sizes[category_index[target_category]], hidden_dim, label_num, num_heads, dropout)
        self.model = HAN(meta_paths, [target_category],
                         config.embedding_size * (len(augmentated_types)) + feature_sizes[category_index[target_category]],
                         hidden_dim, label_num, num_heads, dropout)


        # create category-related mapping
        # self.map_cate = []
        # for i, size in enumerate(feature_sizes):
        #     self.map_cate.append(nn.Linear(config.embedding_size, mapping_size))

        self.category_index = category_index
        self.target_category = target_category



    def forward(self, g_ori, augmentated_features, augmentated_types, augmentated_num, method):

        dealed_augmentated_features = {}
        for aug_type in augmentated_types:
            temp_features = None
            for i in range(augmentated_num):
                if method == "mean":
                    if temp_features is not None:
                        #temp_features = torch.add(temp_features, self.map_cate[self.category_index[aug_type]](augmentated_features[aug_type][i]))
                        temp_features = torch.add(temp_features, augmentated_features[aug_type][i])
                    else:
                        #temp_features = self.map_cate[self.category_index[aug_type]](augmentated_features[aug_type][i])
                        temp_features = augmentated_features[aug_type][i]
                elif method == "concate":
                    if temp_features is not None:
                        #temp_features = torch.cat((temp_features, self.map_cate[self.category_index[aug_type]](augmentated_features[aug_type][i])), dim=-1)
                        temp_features = torch.cat((temp_features, augmentated_features[aug_type][i]), dim=-1)
                    else:
                        #temp_features = self.map_cate[self.category_index[aug_type]](augmentated_features[aug_type][i])
                        temp_features = augmentated_features[aug_type][i]
            if method == "mean":
                temp_features = temp_features / augmentated_num

            dealed_augmentated_features[aug_type] = temp_features

        # deep copy
        g = copy.deepcopy(g_ori)

        #g.nodes[self.target_category].data["h"] = self.map_cate[self.category_index[self.target_category]](g.ndata["h"][self.target_category])

        # concate
        for aug_type in augmentated_types:
            g.nodes[self.target_category].data["h"] = torch.cat((g.ndata["h"][self.target_category], dealed_augmentated_features[aug_type]), dim=-1)
        #g.nodes[self.target_category].data["h"] = feature_tensor_normalize(g.ndata["h"][self.target_category])

        # # # mean
        # # for aug_type in augmentated_types:
        # #     g.nodes[self.target_category].data["h"] = torch.add(g.nodes[self.target_category].data["h"], augmentated_features[aug_type])
        # # g.nodes[self.target_category].data["h"] = g.nodes[self.target_category].data["h"] / (len(augmentated_types)+1)


        logits = self.model(g, g.ndata["h"])[self.target_category]

        return logits



class HAN_AUG_P(nn.Module):

    def __init__(self, config, meta_paths, target_category, hidden_dim, label_num, num_heads, dropout, feature_sizes, mapping_size, category_index, augmentated_types, arg_argmentation_num):
        super().__init__()

        self.model = HAN(meta_paths, [target_category], feature_sizes[category_index[target_category]], hidden_dim, label_num, num_heads, dropout)
        self.category_index = category_index
        self.target_category = target_category



    def forward(self, g_ori, g_aug):

        logits = self.model(g_aug, g_aug.ndata["h"])[self.target_category]

        return logits