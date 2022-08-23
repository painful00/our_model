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
    def __init__(self, g, meta_path_dic, method, config, augmentation_generator, category_index):
        super().__init__()

        self.meta_path_dic = meta_path_dic
        self.g = g
        self.meta_path_reach = {}
        self.method = method
        self.config = config
        self.cvae = augmentation_generator
        self.category_index = category_index

        for meta_path in meta_path_dic:
            self.meta_path_reach[meta_path] = dgl.metapath_reachable_graph(g, meta_path_dic[meta_path])


    def estimate_graphon(self, source_node_type, dest_node_type, adj_matrix):

        # mat = self.meta_path_reach[meta_path].adj(scipy_fmt='coo').toarray()
        # _, graphon = learner.estimate_graphon([mat], self.method, self.config)

        source_node_emb = self.cvae.look_up_table[self.category_index[source_node_type]](self.g.ndata["h"][source_node_type])
        dest_node_emb = self.cvae.look_up_table[self.category_index[dest_node_type]](self.g.ndata["h"][dest_node_type])

        similarity_matrix = torch.matmul(source_node_emb, dest_node_emb.t())
        graphon = F.tanh(similarity_matrix + torch.FloatTensor(adj_matrix)).detach().numpy()
        #print(graphon)
        return graphon


    def generate_graph(self, graphon):
        # num_nodes = node_num
        #
        # generated_graphs = simulator.simulate_graphs(graphon, num_graphs=num_graphs, num_nodes=num_nodes, graph_size="fixed")
        # for ind, item in enumerate(generated_graphs):
        #     generated_graphs[ind] = item + np.eye(node_num)

        noise = np.random.rand(graphon.shape[0], graphon.shape[1])
        graph = graphon - noise
        generated_graph = (graph > 0).astype('float')

        return generated_graph

def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

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
    # intra_path augmentation
    # graphon estimator
    graphons = {}
    for edge in edge_types:
        if (edge_types[edge][0] + "-" + edge_types[edge][1]) not in graphons and (edge_types[edge][1] + "-" + edge_types[edge][0]):
            graphons[edge] = path_augmentation.estimate_graphon(edge_types[edge][0], edge_types[edge][1], g[edge].adj(scipy_fmt='coo').toarray())

    # graph generation
    new_gs = []
    for _ in range(config.argmentation_intra_graph_num):
        hetero_dic = {}
        for edge in graphons:
            matrix = path_augmentation.generate_graph(graphons[edge])
            adj_mat = sparse.coo_matrix(matrix)
            hetero_dic[(edge_types[edge][0], edge_types[edge][0] + "-" + edge_types[edge][1], edge_types[edge][1])] = (adj_mat.row, adj_mat.col)
            hetero_dic[(edge_types[edge][1], edge_types[edge][1] + "-" + edge_types[edge][0], edge_types[edge][0])] = (adj_mat.col, adj_mat.row)
        new_g = dgl.heterograph(hetero_dic)
        new_gs.append(new_g)

    # path-reach graph
    augmentated_graphs = []
    for ind in range(config.argmentation_intra_graph_num):
        for path in config.argmentation_path:
            augmentated_graphs.append(dgl.metapath_reachable_graph(new_gs[ind], path_augmentation.meta_path_dic[path]))

    # augmentation
    hetero_dic = {}
    # for edge in edge_types:
    #     hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item.adj(scipy_fmt='coo').toarray())
        hetero_dic[(target_category, target_category + "-" + target_category + str(ind), target_category + str(ind))] = (adj_mat.row, adj_mat.col)
        hetero_dic[(target_category + str(ind), target_category + str(ind) + "-" + target_category, target_category)] = (adj_mat.col, adj_mat.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in new_g.ntypes:
        if nodetype not in g.ntypes:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category].detach()
        else:
            new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype].detach()


    # for path in config.argmentation_path:
    #     if not os.path.exists("./output/" + path + ".npz"):
    #         graphons[path] = path_augmentation.estimate_graphon(path)
    #         np.savez("./output/" + path + ".npz", graphon=graphons[path])
    #     else:
    #         data = np.load("./output/" + path + ".npz")
    #         graphons[path] = data['graphon']
    #
    # augmentated_graphs = []
    # # intra-path augmentation
    # for path in config.argmentation_path:
    #     augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_intra_graph_num,graphons[path])
    #
    # # inter-path augmentation
    # arg_W = []
    # combinations = list(itertools.combinations(config.argmentation_path, 2))
    # for com1, com2 in combinations:
    #     new_graphon = (graphons[com1] + graphons[com2]) / 2
    #     arg_W.append(new_graphon)
    #     augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,config.argmentation_inter_graph_num,new_graphon)
    #
    # # augmentation
    # hetero_dic = {}
    # for edge in edge_types:
    #     hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    # for ind, item in enumerate(augmentated_graphs):
    #     adj_mat = sparse.coo_matrix(item)
    #     hetero_dic[(target_category, target_category + "-" + target_category + str(ind), target_category + str(ind))] = (adj_mat.row, adj_mat.col)
    #     hetero_dic[(target_category + str(ind), target_category + str(ind) + "-" + target_category, target_category)] = (adj_mat.col, adj_mat.row)
    # new_g = dgl.heterograph(hetero_dic)
    # for nodetype in new_g.ntypes:
    #     if nodetype not in g.ntypes:
    #         new_g.nodes[nodetype].data["h"] = g.ndata["h"][target_category]
    #     else:
    #         new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]


    return new_g, augmentated_graphs
