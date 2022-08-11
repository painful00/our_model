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
from utils import load_data
from openhgnn import HAN

# conf setting
model = "HAN"
dataset = "acm"
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

# data
#g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types = load_data(dataset)

