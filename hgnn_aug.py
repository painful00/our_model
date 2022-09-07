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


# conf setting
model_type = "HPN"
dataset = "yelp"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
argmenter = "STR_META"
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


# augmentation generator
if config.is_augmentation:
    path = "./output/rcvae_"+dataset+".pkl"
    if os.path.exists(path):
        augmentation_generator = VAE(config.embedding_size, config.arg_latent_size, category_index, feature_sizes, e_type_index, has_feature)
        augmentation_generator.load_state_dict(torch.load(path))
    else:
        augmentation_generator = VAE(config.embedding_size, config.arg_latent_size, category_index, feature_sizes, e_type_index, has_feature)
        print("Augmentation generator is not trained")

# Structure Augmentation
augmented_features = {}
if config.is_augmentation:
    for aug_type in config.arg_argmentation_type:
        augmented_features[aug_type] = []
        aug_category = [category_index[aug_type], category_index[target_category]]
        edge_t = None
        for e in edge_types:
            if edge_types[e] == [aug_type, target_category]:
                edge_t = e
                break
        for _ in range(config.arg_argmentation_num):
            z = torch.randn([g.ndata["h"][target_category].size()[0], config.arg_latent_size])
            temp_features = augmentation_generator.inference(z, g.ndata["h"][target_category], aug_category, edge_t).detach()
            augmented_features[aug_type].append(temp_features)

# model
if model_type == "HAN":
    mapping_size = 256
    if config.is_augmentation:
        model = HAN_AUG(config, meta_paths, target_category, config.hidden_dim, label_num, config.num_heads, config.dropout, feature_sizes, mapping_size, category_index, config.arg_argmentation_type, config.arg_argmentation_num)
    else:
        model = HAN(meta_paths, [target_category], target_feature_size, config.hidden_dim, label_num, config.num_heads, config.dropout)
elif model_type == "MAGNN":
    model = MAGNN_AUG(config, label_num, g, dataset, target_category, feature_sizes, category_index)
elif model_type == "SimpleHGN":
    model = SimpleHGN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
elif model_type == "HGT":
    model = HGT_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset)
elif model_type == "HPN":
    model = HPN_AUG(config, g, feature_sizes, category_index, target_category, label_num, dataset, meta_paths)
elif model_type == "CompGCN":
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
method = "mean"
for epoch in range(config.max_epoch):
    model.train()
    if model_type == "HAN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "MAGNN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "SimpleHGN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "HGT":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "HPN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "CompGCN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)

    loss = F.cross_entropy(logits[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[idx_train], labels[idx_train])

    # evaluation
    model.eval()
    with torch.no_grad():
        if model_type == "HAN":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        elif model_type == "MAGNN":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        elif model_type == "SimpleHGN":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        elif model_type == "HGT":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        elif model_type == "HPN":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
        elif model_type == "CompGCN":
            logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
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
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "MAGNN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "SimpleHGN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "HGT":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "HPN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    elif model_type == "CompGCN":
        logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))
