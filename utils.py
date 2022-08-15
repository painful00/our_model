import dgl
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
from sklearn.metrics import f1_score
import datetime

def load_data(dataset_str):
    data_dir = "./data/"+dataset_str+"/graph.bin"
    g, _ = dgl.load_graphs(data_dir)
    g = g[0].long()
    edge_types = {}

    if dataset_str == "acm":
        labels = g.ndata['label']['paper'].to(torch.long)
        idx_train = g.ndata['train_mask']['paper'].squeeze()
        idx_test = g.ndata['test_mask']['paper'].squeeze()
        idx_val = g.ndata['val_mask']['paper'].squeeze()
        for e in g.etypes:
            e1 = e.split('_')[0]
            e2 = e.split('_')[1]
            edge_types[e] = [e1,e2]
        meta_paths = {"PA":['paper_author', 'author_paper'], "PSP":['paper_subject', 'subject_paper']}
        target_category = "paper"
        labels = labels.squeeze()

    elif dataset_str == "dblp":
        labels = g.ndata['labels']['A'].to(torch.long)
        idx_train = g.ndata['train_mask']['A']
        idx_test = g.ndata['test_mask']['A']
        idx_val = g.ndata['val_mask']['A']
        for e in g.etypes:
            e1 = e.split('-')[0]
            e2 = e.split('-')[1]
            edge_types[e] = [e1,e2]
        meta_paths = {"PAP":['A-P', 'P-A'], "APTPA":['A-P', 'P-T', 'T-P', 'P-A'], "APVPA":['A-P', 'P-V', 'V-P', 'P-A']}
        target_category = "A"
        labels = labels.squeeze()

    elif dataset_str == "imdb":
        labels = g.ndata['labels']['M'].to(torch.long)
        idx_train = g.ndata['train_mask']['M']
        idx_test = g.ndata['test_mask']['M']
        idx_val = g.ndata['val_mask']['M']
        for e in g.etypes:
            e1 = e.split('-')[0]
            e2 = e.split('-')[1]
            edge_types[e] = [e1,e2]
        meta_paths = {"MAM":['M-A', 'A-M'], "MDM":['M-D', 'D-M']}
        target_category = "M"
        labels = labels.squeeze()

    idx_train = torch.nonzero(idx_train).squeeze()
    idx_test = torch.nonzero(idx_test).squeeze()
    idx_val = torch.nonzero(idx_val).squeeze()

    category_index = {}
    for ind, type in enumerate(g.ntypes):
        category_index[type] = ind

    feature_sizes = []
    for cat in category_index:
        feature_sizes.append(g.ndata['h'][cat].size()[1])

    return g, idx_train, idx_val, idx_test, labels, category_index, feature_sizes, edge_types, meta_paths, target_category


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = './output/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))