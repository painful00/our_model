import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import rcvae_pretrain
from utils import load_data, feature_tensor_normalize, score, EarlyStopping
import os
from conf import Config
from path_aug_model import Path_Augmentation, path_augmentation_han, path_augmentation_magnn, path_augmentation_simplehgn
import itertools
from tqdm import trange
from scipy import sparse
import dgl
import pickle
from model.HAN import HAN_AUG_P
from model.MAGNN import MAGNN_AUG_P
from model.SimpleHGN import SimpleHGN_AUG_P
from openhgnn import HAN



model_type = "SimpleHGN"
dataset = "imdb"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
config = Config(file_path=conf_path, model=model_type, dataset=dataset, gpu=gpu)


torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

# Load data
g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types, meta_paths, target_category = load_data(dataset)
label_num = int(labels.max()+1)
target_feature_size = g.ndata["h"][target_category].size()[1]
e_type_index = {}
for ind,e in enumerate(g.etypes):
    e_type_index[e] = ind
if dataset == "yelp":
    has_feature = True
    config.embedding_size = feature_sizes[0]
else:
    has_feature = False
if dataset == "acm":
    G = dgl.heterograph({
            ('author', 'author-paper', 'paper'): g['author_paper'].edges(),
            ('paper', 'paper-author', 'author'): g["paper_author"].edges(),
            ('paper', 'paper-subject', 'subject'): g["paper_subject"].edges(),
            ('subject', 'subject-paper', 'paper'): g["subject_paper"].edges()
        })
    G.nodes["paper"].data['label'] = g.ndata["label"]["paper"]
    G.nodes["paper"].data['train_mask'] = g.ndata["train_mask"]["paper"]
    G.nodes["paper"].data['val_mask'] = g.ndata["val_mask"]["paper"]
    G.nodes["paper"].data['test_mask'] = g.ndata["test_mask"]["paper"]
    G.nodes["paper"].data['h'] = g.ndata["h"]["paper"]
    G.nodes["author"].data['h'] = g.ndata["h"]["author"]
    G.nodes["subject"].data['h'] = g.ndata["h"]["subject"]
    g = G


# meta-path augmentation
method = "SGWB"    # ['SBA', 'SAS', 'LG', 'MC', 'USVT', 'GWB', 'SGWB', 'FGWB', 'SFGWB']
path_augmentation = Path_Augmentation(g, meta_paths, method, config)

if model_type == "HAN":
    new_g, augmentated_graphs = path_augmentation_han(g,path_augmentation,config,target_category,edge_types)
elif model_type == "MAGNN":
    new_g, augmentated_graphs = path_augmentation_magnn(g, path_augmentation, config, target_category, edge_types, feature_sizes, category_index)
    category_index = {}
    for ind, type in enumerate(new_g.ntypes):
        category_index[type] = ind
    feature_sizes = []
    for cat in category_index:
        feature_sizes.append(new_g.ndata['h'][cat].size()[1])
elif model_type == "SimpleHGN":
    new_g, augmentated_graphs = path_augmentation_simplehgn(g, path_augmentation, config, target_category, edge_types)


# model
mapping_size = 256
if config.is_augmentation:
    if model_type == "HAN":
        #meta_paths = {}
        for ind, item in enumerate(augmentated_graphs):
            meta_paths["Inter_AUG_"+str(ind)] = ["Inter_AUG_"+str(ind)]
        model = HAN_AUG_P(config, meta_paths, target_category, config.hidden_dim, label_num, config.num_heads, config.dropout, feature_sizes, mapping_size, category_index, config.arg_argmentation_type, config.arg_argmentation_num)
    elif model_type == "MAGNN":
        model = MAGNN_AUG_P(config, label_num, new_g, dataset, target_category, feature_sizes, category_index)
    elif model_type == "SimpleHGN":
        model = SimpleHGN_AUG_P(config, new_g, g, feature_sizes, category_index, target_category, label_num, dataset)

else:
    if model_type == "HAN":
        model = HAN(meta_paths, [target_category], target_feature_size, config.hidden_dim, label_num, config.num_heads, config.dropout)
    elif model_type == "MAGNN":
        model = MAGNN_AUG_P(config, label_num, new_g, dataset, target_category, feature_sizes, category_index)
    elif model_type == "SimpleHGN":
        model = SimpleHGN_AUG_P(config, g, feature_sizes, category_index, target_category, label_num, dataset)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
stopper = EarlyStopping(patience=config.patience)
if gpu >= 0:
    model.to("cuda:"+gpu)
    g.to("cuda:"+gpu)
    idx_train.to("cuda:"+gpu)
    idx_val.to("cuda:"+gpu)
    idx_test.to("cuda:"+gpu)
    labels.to("cuda:"+gpu)


# train
for epoch in range(config.max_epoch):
    model.train()
    if model_type == "HAN":
        logits = model(g, new_g)
    elif model_type == "MAGNN":
        logits = model(new_g)
    elif model_type == "SimpleHGN":
        logits = model(new_g, g)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[idx_train], labels[idx_train])

    # evaluation
    model.eval()
    with torch.no_grad():
        if model_type == "HAN":
            logits = model(g, new_g)
        elif model_type == "MAGNN":
            logits = model(new_g)
        elif model_type == "SimpleHGN":
            logits = model(new_g, g)
    val_loss = F.cross_entropy(logits[idx_val], labels[idx_val])
    val_acc, val_micro_f1, val_macro_f1 = score(logits[idx_val], labels[idx_val])

    early_stop = stopper.step(val_loss.data.item(), val_acc, model)

    print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
          'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

    test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
    test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1,
                                                                                  test_macro_f1))

    if early_stop:
        break

stopper.load_checkpoint(model)
model.eval()
with torch.no_grad():
    if model_type == "HAN":
        logits = model(g, new_g)
    elif model_type == "MAGNN":
        logits = model(new_g)
    elif model_type == "SimpleHGN":
        logits = model(new_g, g)
test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))


