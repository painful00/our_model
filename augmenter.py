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
    if os.path.exists():
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
