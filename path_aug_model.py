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
from conf import Config
import dgl
from tqdm import trange
from typing import List, Tuple
from model import learner
from model import simulator
from scipy import sparse



class Path_Augmentation:
    def __init__(self, g, meta_path_dic, method, config):
        super().__init__()

        self.meta_path_dic = meta_path_dic
        self.g = g
        self.meta_path_reach = {}
        self.method = method
        self.config = config

        for meta_path in meta_path_dic:
            self.meta_path_reach[meta_path] = dgl.metapath_reachable_graph(g, meta_path_dic[meta_path])


    def estimate_graphon(self, meta_path):

        mat = self.meta_path_reach[meta_path].adj(scipy_fmt='coo').toarray()
        _, graphon = learner.estimate_graphon([mat], self.method, self.config)

        return graphon

    def generate_graph(self, node_num, num_graphs, graphon):
        num_nodes = node_num

        generated_graphs = simulator.simulate_graphs(graphon, num_graphs=num_graphs, num_nodes=num_nodes, graph_size="fixed")
        for ind, item in enumerate(generated_graphs):
            generated_graphs[ind] = item + np.eye(node_num)

        return generated_graphs



