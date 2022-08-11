import dgl
import numpy as np
import torch
import torch.nn.functional as F


def load_data(dataset_str):
    data_dir = "./data/"+dataset_str+"/graph.bin"
    g, _ = dgl.load_graphs(data_dir)
    g = g[0].long()
    edge_types = {}

    if dataset_str == "acm":
        labels = g.ndata['label']['paper']
        idx_train = g.ndata['train_mask']['paper']
        idx_test = g.ndata['test_mask']['paper']
        idx_val = g.ndata['val_mask']['paper']
        for e in g.etypes:
            e1 = e.split('_')[0]
            e2 = e.split('_')[1]
            edge_types[e] = [e1,e2]
    if dataset_str == "dblp":
        labels = g.ndata['labels']['A']
        idx_train = g.ndata['train_mask']['A']
        idx_test = g.ndata['test_mask']['A']
        idx_val = g.ndata['val_mask']['A']
        for e in g.etypes:
            e1 = e.split('-')[0]
            e2 = e.split('-')[1]
            edge_types[e] = [e1,e2]
    if dataset_str == "imdb":
        labels = g.ndata['labels']['M']
        idx_train = g.ndata['train_mask']['M']
        idx_test = g.ndata['test_mask']['M']
        idx_val = g.ndata['val_mask']['M']
        for e in g.etypes:
            e1 = e.split('-')[0]
            e2 = e.split('-')[1]
            edge_types[e] = [e1,e2]

    idx_train = torch.nonzero(idx_train).squeeze()
    idx_test = torch.nonzero(idx_test).squeeze()
    idx_val = torch.nonzero(idx_val).squeeze()

    category_index = {}
    for ind, type in enumerate(g.ntypes):
        category_index[type] = ind

    feature_sizes = []
    for cat in category_index:
        feature_sizes.append(g.ndata['h'][cat].size()[1])

    for node_type in g.ntypes:
        g.ndata["h"][node_type] = feature_tensor_normalize(g.ndata["h"][node_type])

    return g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature