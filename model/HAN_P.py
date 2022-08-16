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

        self.model = HAN(meta_paths, [target_category], feature_sizes[category_index[target_category]], hidden_dim, label_num, num_heads, dropout)
        self.category_index = category_index
        self.target_category = target_category



    def forward(self, g_ori, g_aug):

        logits = self.model(g_aug, g_aug.ndata["h"])[self.target_category]

        return logits