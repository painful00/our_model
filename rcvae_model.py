import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, embedding_size, latent_size, category_index, feature_sizes, edge_type_index, has_feature):
        super().__init__()

        assert type(latent_size) == int
        assert type(embedding_size) == int
        assert type(category_index) == dict
        assert type(feature_sizes) == list
        assert type(edge_type_index) == dict

        self.latent_size = latent_size
        self.category_index = category_index
        self.feature_sizes = feature_sizes
        self.look_up_table = []

        # category-related map
        if has_feature:
            for i, size in enumerate(feature_sizes):
                self.look_up_table.append(self.identical_map)
            self.embedding_size = feature_sizes[0]
        else:
            for i, size in enumerate(feature_sizes):
                self.look_up_table.append(nn.Linear(size, embedding_size))
            self.embedding_size = embedding_size

        self.encoder = Encoder([self.embedding_size, latent_size], latent_size, category_index, edge_type_index)
        self.decoder = Decoder([self.embedding_size, self.embedding_size], category_index, latent_size, edge_type_index)


    def forward(self, x, c, category, edge_type):

        x_emb = self.look_up_table[category[0]](x)
        c_emb = self.look_up_table[category[1]](c)
        means, log_var = self.encoder(x_emb, c_emb, edge_type)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c_emb, edge_type)
        return recon_x, means, log_var, z, x_emb

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c, category, edge_type):

        c_emb = self.look_up_table[category[1]](c)
        recon_x = self.decoder(z, c_emb, edge_type)

        return recon_x

    def identical_map(self, x):
        return x

    def find_embedding(self, x, category):
        return self.look_up_table[self.category_index[category]](x).detach()

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, category_index, edge_type_index):
        super().__init__()

        self.edge_type_index = edge_type_index
        self.category_index = category_index
        self.MLP = nn.Sequential()
        layer_sizes[0] = layer_sizes[0] * 2 + len(self.edge_type_index)

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c, edge_type):

        x = torch.cat((x, c), dim=-1)
        edge_embedding = F.one_hot(torch.LongTensor([self.edge_type_index[edge_type]]), len(self.edge_type_index)).repeat(x.size()[0], 1)
        x = torch.cat((x, edge_embedding), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, category_index, latent_size, edge_type_index):

        super().__init__()

        self.MLP = nn.Sequential()
        self.category_index = category_index
        self.edge_type_index = edge_type_index
        layer_sizes[0] = layer_sizes[0] + latent_size + len(edge_type_index)
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c, edge_type):

        z = torch.cat((z, c), dim=-1)
        edge_embedding = F.one_hot(torch.LongTensor([self.edge_type_index[edge_type]]), len(self.edge_type_index)).repeat(z.size()[0], 1)
        z = torch.cat((z, edge_embedding), dim=-1)
        x = self.MLP(z)
        rec_x = F.sigmoid(x)

        return rec_x

