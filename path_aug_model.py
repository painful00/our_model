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
import itertools



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



def path_augmentation_han(g,path_augmentation,config,target_category,edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   config.argmentation_intra_graph_num,
                                                                                   graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   config.argmentation_inter_graph_num,
                                                                                   new_graphon)

    # augmentation
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, "AUG_" + str(ind), target_category)] = (adj_mat.row, adj_mat.col)
    new_g = dgl.heterograph(hetero_dic)
    new_g.nodes[target_category].data["h"] = g.ndata["h"][target_category]

    return new_g, augmentated_graphs


def path_augmentation_magnn(g, path_augmentation, config, target_category, edge_types, feature_sizes, category_index):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_intra_graph_num,graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_inter_graph_num,new_graphon)

    # augmentation
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, target_category+"-"+target_category+str(ind), target_category+str(ind))] = (adj_mat.row, adj_mat.col)
        hetero_dic[(target_category+str(ind), target_category+str(ind)+"-"+target_category, target_category)] = (adj_mat.col, adj_mat.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]

    return new_g, augmentated_graphs



def path_augmentation_simplehgn(g, path_augmentation, config, target_category, edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_intra_graph_num,graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_inter_graph_num,new_graphon)

    # augmentation
    hetero_dic = {}
    # for edge in edge_types:
    #     hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, target_category + "-" + target_category + str(ind), target_category + str(ind))] = (adj_mat.row, adj_mat.col)
        hetero_dic[(target_category + str(ind), target_category + str(ind) + "-" + target_category, target_category)] = (adj_mat.col, adj_mat.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]


    return new_g, augmentated_graphs


def path_augmentation_hgt(g, path_augmentation, config, target_category, edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_intra_graph_num,graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_inter_graph_num,new_graphon)

    # augmentation
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, target_category + "-" + target_category + str(ind), target_category + str(ind))] = (adj_mat.row, adj_mat.col)
        hetero_dic[(target_category + str(ind), target_category + str(ind) + "-" + target_category, target_category)] = (adj_mat.col, adj_mat.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]


    return new_g, augmentated_graphs


def path_augmentation_hpn(g,path_augmentation,config,target_category,edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   config.argmentation_intra_graph_num,
                                                                                   graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   config.argmentation_inter_graph_num,
                                                                                   new_graphon)

    # augmentation
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, "AUG_" + str(ind), target_category)] = (adj_mat.row, adj_mat.col)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]
    #new_g.nodes[target_category].data["h"] = g.ndata["h"][target_category]


    return new_g, augmentated_graphs


def path_augmentation_compgcn(g, path_augmentation, config, target_category, edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata["h"][target_category].size()[0]
    for path in config.argmentation_path:
        if not os.path.exists("./output/" + path + ".npz"):
            graphons[path] = path_augmentation.estimate_graphon(path)
            np.savez("./output/" + path + ".npz", graphon=graphons[path])
        else:
            data = np.load("./output/" + path + ".npz")
            graphons[path] = data['graphon']

    augmentated_graphs = []
    # intra-path augmentation
    for path in config.argmentation_path:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_intra_graph_num,graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(config.argmentation_path, 2))
    for com1, com2 in combinations:
        new_graphon = (graphons[com1] + graphons[com2]) / 2
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_inter_graph_num,new_graphon)

    # augmentation
    hetero_dic = {}
    # for edge in edge_types:
    #     hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, target_category + "-" + target_category + str(ind), target_category + str(ind))] = (adj_mat.row, adj_mat.col)
        hetero_dic[(target_category + str(ind), target_category + str(ind) + "-" + target_category, target_category)] = (adj_mat.col, adj_mat.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]


    return new_g, augmentated_graphs