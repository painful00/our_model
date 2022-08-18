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
from conf import Config

from tqdm import trange

dataset = "yelp"
gpu = -1    #   -1:cpu    >0:gpu
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "conf.ini")
conf_path = os.path.abspath(configPath)
config = Config(file_path=conf_path, model=None, dataset=dataset, gpu=gpu)

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)


# Load data
g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types, meta_paths, target_category = load_data(dataset)

# Pretrain
best_augmented_features = None

best_augmented_features = rcvae_pretrain.generated_generator(config, config.device, g, category_index, feature_sizes, edge_types, dataset)
