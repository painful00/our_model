import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from rcvae_model import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import copy
import dgl


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    #BCE = torch.nn.CosineEmbeddingLoss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)


def generated_generator(config, device, g, category_index, feature_sizes, edge_types, dataset):
    x_list, c_list, category_list = [], [], []
    # edge_types:{edge_type:[head_node_type,tail_node_type],...}
    for edge in edge_types:
        x_type = edge_types[edge][0]
        c_type = edge_types[edge][1]
        x_id_list, c_id_list = g[edge].edges()
        x_list.append(g.ndata["h"][x_type][x_id_list])
        c_list.append(g.ndata["h"][c_type][c_id_list])
        category_list.append([x_type, c_type])

    rcvae_dataset_dataloaders = []
    for i, edge in enumerate(edge_types):
        rcvae_dataset = TensorDataset(x_list[i], c_list[i])
        rcvae_dataset_sampler = RandomSampler(rcvae_dataset)
        rcvae_dataset_dataloader = DataLoader(rcvae_dataset, sampler=rcvae_dataset_sampler, batch_size=config.arg_batch_size)
        rcvae_dataset_dataloaders.append(rcvae_dataset_dataloader)

    gc.collect()

    # Pretrain
    rcvae = VAE(embedding_size=config.embedding_size,
               latent_size=config.arg_latent_size,
               category_index=category_index,
               feature_sizes=feature_sizes)
    rcvae_optimizer = optim.Adam(rcvae.parameters(), lr=config.arg_pretrain_lr)

    if torch.cuda.is_available():
        rcvae.to(config.device)

    # Pretrain
    best_augmented_features = None
    rcvae_model = None
    for epoch in trange(config.arg_pretrain_epochs, desc='Run CVAE Train'):
        print("************",epoch,"*************")
        for i in trange(len(rcvae_dataset_dataloaders), desc='Run Edges'):
            rcvae_dataset_dataloader = rcvae_dataset_dataloaders[i]
            category = category_list[i]
            for _, (x, c) in enumerate(tqdm(rcvae_dataset_dataloader)):
                rcvae.train()
                if torch.cuda.is_available():
                    x, c = x.to(device), c.to(device)

                recon_x, mean, log_var, _, x = rcvae(x, c, [category_index[category[0]], category_index[category[1]]])
                rcvae_loss = loss_fn(recon_x, x, mean, log_var)
                rcvae_optimizer.zero_grad()
                rcvae_loss.backward()
                rcvae_optimizer.step()

            print("loss: ", rcvae_loss)

    torch.save(rcvae.state_dict(), "./output/rcvae_"+ dataset+".pkl")




                # augmentation
                # z = torch.randn([feature_sizes[category_index[category[1]]], args.latent_size])
                # if args.cuda:
                #     z = z.to(device)
                # augmented_features = rcvae.inference(z, cvae_features)
                # augmented_features = feature_tensor_normalize(augmented_features).detach()



    return best_augmented_features