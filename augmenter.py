import dgl
import numpy as np
import torch
import torch.nn.functional as F
import copy
import gc
from scipy import sparse
import itertools
from rcvae_pretrain import loss_fn
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from cvae_model import VAE
import torch.optim as optim
import os
import random



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


def LA(g, augmentation_generator, arg_argmentation_num, target_category, arg_latent_size, edge_types):

    augmented_features = []

    for _ in range(arg_argmentation_num):
        z = torch.randn([g.ndata["h"][target_category].size()[0], arg_latent_size])
        temp_features = augmentation_generator.inference(z, g.ndata["h"][target_category]).detach()
        augmented_features.append(temp_features)

    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in g.ntypes:
        new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]
        if nodetype == target_category:
            for item in augmented_features:
                new_g.nodes[nodetype].data["h"] = torch.cat((new_g.ndata["h"][target_category], item), dim=-1)

    return new_g

def LA_Train(config, device, g, category_index, feature_sizes, edge_types, dataset):
    x_list, c_list = [], []
    # edge_types:{edge_type:[head_node_type,tail_node_type],...}

    dimension = max(feature_sizes)
    for edge in edge_types:
        x_type = edge_types[edge][0]
        c_type = edge_types[edge][1]
        x_id_list, c_id_list = g[edge].edges()
        dimension_x = g.ndata["h"][x_type].size()[1]
        dimension_c = g.ndata["h"][c_type].size()[1]
        if dimension_x < dimension:
            num = g.ndata["h"][x_type][x_id_list].size()[0]
            x_list.append(torch.cat((g.ndata["h"][x_type][x_id_list], torch.zeros(num, dimension - dimension_x)), dim=-1))
        else:
            x_list.append(g.ndata["h"][x_type][x_id_list])

        if dimension_c < dimension:
            num = g.ndata["h"][c_type][c_id_list].size()[0]
            c_list.append(torch.cat((g.ndata["h"][c_type][c_id_list], torch.zeros(num, dimension - dimension_c)), dim=-1))
        else:
            c_list.append(g.ndata["h"][c_type][c_id_list])

    cvae_dataset_dataloaders = []
    for i, edge in enumerate(edge_types):
        cvae_dataset = TensorDataset(x_list[i], c_list[i])
        cvae_dataset_sampler = RandomSampler(cvae_dataset)
        cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler,
                                              batch_size=config.arg_batch_size)
        cvae_dataset_dataloaders.append(cvae_dataset_dataloader)

    gc.collect()

    # Pretrain
    cvae = VAE(embedding_size=config.embedding_size,
                latent_size=config.arg_latent_size,
                feature_sizes=feature_sizes)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=config.arg_pretrain_lr)

    if torch.cuda.is_available():
        cvae.to(config.device)

    # Pretrain
    if os.path.exists("./output/cvae_" + dataset + ".pkl"):
        cvae = VAE(embedding_size=config.embedding_size,
                   latent_size=config.arg_latent_size,
                   feature_sizes=feature_sizes)
        cvae.load_state_dict(torch.load("./output/cvae_" + dataset + ".pkl"))
    else:
        for epoch in trange(config.arg_pretrain_epochs, desc='Run CVAE Train'):
            print("************", epoch, "*************")
            for i in trange(len(cvae_dataset_dataloaders), desc='Run Edges'):
                cvae_dataset_dataloader = cvae_dataset_dataloaders[i]
                for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)):
                    cvae.train()
                    if torch.cuda.is_available():
                        x, c = x.to(device), c.to(device)

                    recon_x, mean, log_var, _, x = cvae(x, c)
                    cvae_loss = loss_fn(recon_x, x, mean, log_var)
                    cvae_optimizer.zero_grad()
                    cvae_loss.backward()
                    cvae_optimizer.step()

                print("loss: ", cvae_loss)

        torch.save(cvae.state_dict(), "./output/cvae_" + dataset + ".pkl")

    return cvae

def NodeAug(g, target_category, meta_path_dic, edge_types, node_id):
    # replace attribute
    feature = g.ndata["h"][target_category][node_id]
    w_max = feature.max()
    w_ave = feature.mean()
    p_rem = 0.3 * (w_max - feature) / (w_max - w_ave)
    p_nor = torch.ones(p_rem.size())
    p_rem = torch.where(p_rem > p_nor, p_nor, p_rem)

    neighbor_nodes = []
    for meta_path in meta_path_dic:
        adj_row = dgl.metapath_reachable_graph(g, meta_path_dic[meta_path]).adj(scipy_fmt='coo').toarray()[node_id]
        neighbor_nodes = neighbor_nodes + list(np.where(adj_row)[0])
    neighbor_nodes = list(set(neighbor_nodes))
    x_sam = g.ndata["h"][target_category][neighbor_nodes]
    div = x_sam.mean(dim=0) - x_sam.min(dim=0).values
    s_min = x_sam.min(dim=0).values.repeat(x_sam.size()[0], 1)
    p_sam = (x_sam - s_min) / div.repeat(x_sam.size()[0], 1)
    p_sam = F.softmax(torch.where(p_sam != p_sam, 0, p_sam), dim=0)

    feature_sample = []
    for i in range(p_sam.size()[1]):
        feature_sample.append(np.random.choice(x_sam[:, i], p=p_sam[:, i].numpy()))
    feature_sample = torch.Tensor(np.array(feature_sample))

    replace_p = torch.rand(p_rem.size())
    feature_nor = torch.where(p_rem > replace_p, feature_sample, feature)

    # modify edges
    degree = {}
    for edge in edge_types:
        node_type1 = edge_types[edge][0]
        node_type2 = edge_types[edge][1]
        adj = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())
        if node_type1 not in degree:
            degree[node_type1] = adj.sum(dim=1)
        else:
            degree[node_type1] = degree[node_type1] + adj.sum(dim=1)
        if node_type2 not in degree:
            degree[node_type2] = adj.sum(dim=0)
        else:
            degree[node_type2] = degree[node_type2] + adj.sum(dim=0)
    for node_type in degree:
        degree[node_type] = degree[node_type] / 2

    adj_dealed = {}
    for edge in edge_types:
        node_type1 = edge_types[edge][0]
        node_type2 = edge_types[edge][1]

        # remove edges
        if node_type1 == target_category:
            adj_row = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())[node_id]
            source_degree = degree[target_category][node_id].repeat(adj_row.size())
            dest_degree = degree[node_type2]
            s_e = torch.log(torch.cat((source_degree.view(1, -1), dest_degree.view(1, -1)), dim=0).min(dim=0).values)
            s_e = torch.where(adj_row > 0, s_e, adj_row)
            s_e_max = s_e.max()
            s_e_avg = s_e.mean()
            p_rem = (s_e_max - s_e) / (s_e_max - s_e_avg) * 0.3
            p_nor = torch.ones(p_rem.size())
            p_rem = torch.where(p_rem > p_nor, p_nor, p_rem)

            p_per = torch.rand(p_rem.size())
            remove_edges = torch.where(p_rem > p_per, 1, 0)
            remove_edges = torch.where(adj_row > 0, remove_edges, 0)

        # add edges
            adj_row_all = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())[neighbor_nodes].sum(dim=0)
            adj_row_all = torch.where(adj_row_all > 0, 1, 0)
            adj_row_all = torch.where(adj_row>0, 0, adj_row_all)
            s_n = torch.log(dest_degree)
            s_n = torch.where(adj_row_all>0, s_n, 0)
            s_n_min = s_n.min()
            s_n_avg = s_n.mean()
            p_add = (s_n - s_n_min) / (s_n_avg - s_n_min) * 0.3 / 2
            p_nor = torch.ones(p_add.size())
            p_add = torch.where(p_add > p_nor, p_nor, p_add)

            p_per = torch.rand(p_add.size())
            add_edges = torch.where(p_add > p_per, 1, 0)
            add_edges = torch.where(adj_row_all > 0, add_edges, 0)

            adj_dealed[edge] = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())[node_id] + add_edges - remove_edges

    hetero_dic = {}
    for edge in edge_types:
        node_type1 = edge_types[edge][0]
        node_type2 = edge_types[edge][1]

        if node_type1 == target_category:
            adj = torch.FloatTensor(g[edge].adj(scipy_fmt='coo').toarray())
            adj[node_id] = adj_dealed[edge]
            adj_new = sparse.coo_matrix(adj)
            hetero_dic[(node_type1, node_type1 + "-" + node_type2, node_type2)] = (adj_new.row, adj_new.col)
            hetero_dic[(node_type2, node_type2 + "-" + node_type1, node_type1)] = (adj_new.col, adj_new.row)
    new_g = dgl.heterograph(hetero_dic)
    for nodetype in g.ntypes:
        new_g.nodes[nodetype].data["h"] = g.ndata["h"][nodetype]
        if nodetype == target_category:
            new_g.nodes[nodetype].data["h"][node_id] = feature_nor

    return new_g




