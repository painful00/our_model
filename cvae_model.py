import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, embedding_size, latent_size, feature_sizes):
        super().__init__()

        assert type(latent_size) == int
        assert type(embedding_size) == int
        assert type(feature_sizes) == list

        self.latent_size = latent_size
        self.feature_sizes = feature_sizes
        self.look_up_table = []
        self.dimension = max(feature_sizes)
        self.look_up_table = nn.Linear(self.dimension, embedding_size)


        self.encoder = Encoder([embedding_size, latent_size], latent_size)
        self.decoder = Decoder([embedding_size, embedding_size], latent_size)


    def forward(self, x, c):

        x_emb = F.sigmoid(self.look_up_table(x))
        c_emb = F.sigmoid(self.look_up_table(c))
        means, log_var = self.encoder(x_emb, c_emb)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c_emb)
        return recon_x, means, log_var, z, x_emb

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c):

        if c.size()[1] < self.dimension:
            c = torch.cat((c, torch.zeros(c.size()[0], self.dimension-c.size()[1])), dim=-1)

        c_emb = F.sigmoid(self.look_up_table(c))
        recon_x = self.decoder(z, c_emb)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()
        layer_sizes[0] = layer_sizes[0] * 2

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c):

        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        layer_sizes[0] = layer_sizes[0] + latent_size
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c):

        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        rec_x = F.sigmoid(x)

        return rec_x

