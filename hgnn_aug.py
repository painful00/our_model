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
from openhgnn import HAN
from rcvae_model import VAE
from model.HAN import HAN_AUG


# conf setting
model = "HAN"
dataset = "imdb"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
config = Config(file_path=conf_path, model=model, dataset=dataset, gpu=gpu)


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

# augmentation generator
path = "./output/rcvae_"+dataset+".pkl"
if os.path.exists(path):
    augmentation_generator = VAE(config.embedding_size, config.arg_latent_size, category_index, feature_sizes)
    augmentation_generator.load_state_dict(torch.load(path))
else:
    augmentation_generator = VAE(config.arg_encoder_layer_sizes, config.arg_latent_size, config.arg_decoder_layer_sizes,
                                 category_index, feature_sizes)
    print("Augmentation generator is not trained")

# Structure Augmentation
if config.is_augmentation:
    augmented_features = {}
    for aug_type in config.arg_argmentation_type:
        augmented_features[aug_type] = []
        aug_category = [category_index[aug_type], category_index[target_category]]
        for _ in range(config.arg_argmentation_num):
            z = torch.randn([g.ndata["h"][target_category].size()[0], config.arg_latent_size])
            temp_features = augmentation_generator.inference(z, g.ndata["h"][target_category], aug_category).detach()
            augmented_features[aug_type].append(temp_features)


# model
mapping_size = 256
if config.is_augmentation:
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
method = "mean"
for epoch in range(config.max_epoch):
    model.train()
    logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[idx_train], labels[idx_train])

    # evaluation
    model.eval()
    with torch.no_grad():
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
    logits = model(g, augmented_features, config.arg_argmentation_type, config.arg_argmentation_num, method)
test_loss = F.cross_entropy(logits[idx_test], labels[idx_test])
test_acc, test_micro_f1, test_macro_f1 = score(logits[idx_test], labels[idx_test])
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_micro_f1, test_macro_f1))
