import dgl
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy import sparse
import itertools

def DropEdge(rate, g, edge_types, target_category):
    hetero_dic = {}
    deal_edges = []
    for edge in edge_types:
        node_type1 = edge.split("-")[0]
        node_type2 = edge.split("-")[1]
        if (node_type1 + "-" + node_type2) not in hetero_dic and (node_type2 + "-" + node_type1) not in hetero_dic:
            deal_edges.append(node_type1 + "-" + node_type2)
            deal_edges.append(node_type2 + "-" + node_type1)
            adj_ori = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())
            adj_drop = F.dropout(adj_ori, rate) * (1-rate)
            adj_new = sparse.coo_matrix(adj_drop)
            hetero_dic[(node_type1, node_type1 + "-" + node_type2, node_type2)] = (adj_new.row, adj_new.col)
            hetero_dic[(node_type2, node_type2 + "-" + node_type1, node_type1)] = (adj_new.col, adj_new.row)
    node_dic = {}
    for nodetype in g.ntypes:
        node_dic[nodetype] = g.num_nodes(nodetype)
    new_g = dgl.heterograph(hetero_dic, node_dic)
    for nodetype in g.ntypes:
        new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]
    return new_g
