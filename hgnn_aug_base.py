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
from tqdm import trange
from conf import Config
import dgl
from utils import load_data, score, EarlyStopping
from openhgnn import HAN, MAGNN
from openhgnn.models.MAGNN import mp_instance_sampler
from rcvae_model import VAE
from model.HAN import HAN_AUG
from model.MAGNN import MAGNN_AUG
from model.SimpleHGN import SimpleHGN_AUG
from model.HGT import HGT_AUG
from model.HPN import HPN_AUG
from model.CompGCN import CompGCN_AUG
from augmenter import DropEdge, LA, LA_Train, NodeAug, NASA, NASA_Loss


# conf setting
model_type = "HAN"
dataset = "imdb"
argmenter = "NASA"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
config = Config(file_path=conf_path, model=model_type, dataset=dataset, gpu=gpu, augmenter=argmenter)

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
dgl.seed(config.seed)

# data loading
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
    meta_paths = {"PAP":['paper-author', 'author-paper'], "PSP":['paper-subject', 'subject-paper']}
    edge_types = {}
    for e in g.etypes:
        e1 = e.split('-')[0]
        e2 = e.split('-')[1]
        edge_types[e] = [e1, e2]

# augmented graph feature
if argmenter == "LA":
    feature_generator = LA_Train(config, config.device, g, category_index, feature_sizes, edge_types, dataset)
    new_g = LA(g, feature_generator, config.arg_argmentation_num, target_category, config.arg_latent_size, edge_types)
    target_feature_size = new_g.ndata["h"][target_category].size()[1]
    feature_sizes = []
    for cat in category_index:
        feature_sizes.append(new_g.ndata['h'][cat].size()[1])



# model
if model_type == "HAN":
    mapping_size = 256
    if config.is_augmentation:
        model = HAN_AUG(config, meta_paths, target_category, config.hidden_dim, label_num, config.num_heads, config.dropout, feature_sizes, mapping_size, category_index, config.arg_argmentation_type, config.arg_argmentation_num)
    else:
        model = HAN(meta_paths, [target_category], target_feature_size, config.hidden_dim, label_num, config.num_heads, config.dropout)
elif model_type == "MAGNN":
    if argmenter == "LA":
        config.is_augmentation=True
        model = MAGNN_AUG(config, label_num, g, dataset, target_category, feature_sizes, category_index)
        config.is_augmentation = False
    else:
        model = MAGNN_AUG(config, label_num, g, dataset, target_category, feature_sizes, category_index)
elif model_type == "SimpleHGN":
    if argmenter == "LA":
        config.is_augmentation = True
        model = SimpleHGN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
        config.is_augmentation = False
    else:
        model = SimpleHGN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
elif model_type == "HGT":
    if argmenter == "LA":
        config.is_augmentation = True
        model = HGT_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
        config.is_augmentation = False
    else:
        model = HGT_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
elif model_type == "HPN":
    if argmenter == "LA":
        config.is_augmentation = True
        model = HPN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset, meta_paths)
        config.is_augmentation = False
    else:
        model = HPN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset, meta_paths)
elif model_type == "CompGCN":
    if argmenter == "LA":
        config.is_augmentation = True
        model = CompGCN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
        config.is_augmentation = False
    else:
        model = CompGCN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
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
augmented_features = []
method = "mean"
for epoch in range(config.max_epoch):
    model.train()
    if argmenter == "dropedge":
        new_g = DropEdge(config.dropedge_rate, g, edge_types, target_category)
    elif argmenter == "LA":
        new_g = LA(g, feature_generator, config.arg_argmentation_num, target_category, config.arg_latent_size,edge_types)
    elif argmenter == "NodeAug":
        perturb_id = random.randint(0, g.num_nodes(target_category)-1)
        new_g = NodeAug(g, target_category, meta_paths, edge_types, perturb_id)
    elif argmenter == "NASA":
        perturb_id = random.randint(0, g.num_nodes(target_category) - 1)
        new_g = NASA(g, target_category, meta_paths, edge_types, perturb_id)

    if argmenter == "NodeAug":
        if model_type == "HAN":
            logits1 = model(new_g, new_g.ndata["h"])[target_category]
            with torch.no_grad():
                logits2 = model(g, g.ndata["h"])[target_category]
            logits = model(g, g.ndata["h"])[target_category]
        else:
            logits1 = model(new_g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
            with torch.no_grad():
                logits2 = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train]) + F.kl_div(F.log_softmax(logits1[perturb_id], dim=-1), F.softmax(logits2[perturb_id], dim=-1))

    elif argmenter == "NASA":
        if model_type == "HAN":
            logits = model(new_g, new_g.ndata["h"])[target_category]
        else:
            logits = model(new_g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        loss_sup = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_cr = NASA_Loss(g, logits, idx_train, idx_val, idx_test, meta_paths)
        loss = loss_sup + 0.1 * loss_cr
    else:
        if model_type == "HAN":
            logits = model(new_g, new_g.ndata["h"])[target_category]
        else:
            logits = model(new_g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[idx_train], labels[idx_train])

    # evaluation
    model.eval()
    with torch.no_grad():
        if argmenter == "NodeAug":
            if model_type == "HAN":
                logits = model(g, g.ndata["h"])[target_category]
            else:
                logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        else:
            if model_type == "HAN":
                logits = model(new_g, new_g.ndata["h"])[target_category]
            else:
                logits = model(new_g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
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
    if argmenter == "NodeAug":
        if model_type == "HAN":
            logits = model(g, g.ndata["h"])[target_category]
        else:
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    else:
        if model_type == "HAN":
            logits = model(new_g, new_g.ndata["h"])[target_category]
        else:
            logits = model(new_g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))
