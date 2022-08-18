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
from path_aug_model import Path_Augmentation
import itertools
from tqdm import trange
from scipy import sparse
import dgl
from model.HAN_P import HAN_AUG
from openhgnn import HAN

model = "HAN"
dataset = "yelp"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
config = Config(file_path=conf_path, model=model, dataset=dataset, gpu=gpu)


torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

# Load data
g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types, meta_paths, target_category = load_data(dataset)
label_num = int(labels.max()+1)
target_feature_size = g.ndata["h"][target_category].size()[1]

# meta-path augmentation
method = "SGWB"    # ['SBA', 'SAS', 'LG', 'MC', 'USVT', 'GWB', 'SGWB', 'FGWB', 'SFGWB']
path_augmentation = Path_Augmentation(g, meta_paths, method, config)

# graphon estimator
graphons = {}
node_num = g.ndata["h"][target_category].size()[0]
for path in config.argmentation_path:
    if not os.path.exists("./output/"+path+".npz"):
        graphons[path] = path_augmentation.estimate_graphon(path)
        np.savez("./output/" + path + ".npz", graphon=graphons[path])
    else:
        data = np.load("./output/" + path + ".npz")
        graphons[path] = data['graphon']


augmentated_graphs = []
# intra-path augmentation
for path in config.argmentation_path:
    augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num, config.argmentation_intra_graph_num, graphons[path])

# inter-path augmentation
arg_W = []
combinations = list(itertools.combinations(config.argmentation_path,2))
for com1, com2 in combinations:
    new_graphon = (graphons[com1]+graphons[com2])/2
    arg_W.append(new_graphon)
    augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num, config.argmentation_inter_graph_num, new_graphon)



# augmentation
hetero_dic = {}
for edge in edge_types:
    hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
for ind, item in enumerate(augmentated_graphs):
    adj_mat = sparse.coo_matrix(item)
    hetero_dic[(target_category,"AUG_"+str(ind),target_category)] = (adj_mat.row, adj_mat.col)
new_g = dgl.heterograph(hetero_dic)
new_g.nodes[target_category].data["h"] = g.ndata["h"][target_category]



# augmentation
hetero_dic = {}
for edge in edge_types:
    hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
for ind, item in enumerate(augmentated_graphs):
    adj_mat = sparse.coo_matrix(item)
    hetero_dic[(target_category,"Inter_AUG_"+str(ind),target_category)] = (adj_mat.row, adj_mat.col)
new_g = dgl.heterograph(hetero_dic)
new_g.nodes[target_category].data["h"] = g.ndata["h"][target_category]


# model
mapping_size = 256
if config.is_augmentation:
    #meta_paths = {}
    for ind, item in enumerate(augmentated_graphs):
        meta_paths["Inter_AUG_"+str(ind)] = ["Inter_AUG_"+str(ind)]
    model = HAN_AUG(config, meta_paths, target_category, config.hidden_dim, label_num, config.num_heads, config.dropout, feature_sizes, mapping_size, category_index, config.arg_argmentation_type, config.arg_argmentation_num)
else:
    model = HAN(meta_paths, [target_category], target_feature_size, config.hidden_dim, label_num, config.num_heads, config.dropout)
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
    logits = model(g, new_g)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[idx_train], labels[idx_train])

    # evaluation
    model.eval()
    with torch.no_grad():
        logits = model(g, new_g)
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
    logits = model(g, new_g)
test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))


