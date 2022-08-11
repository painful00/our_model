import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, category_index, feature_sizes):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        assert type(category_index) == dict
        assert type(feature_sizes) == list

        self.latent_size = latent_size
        self.category_index = category_index
        self.feature_sizes = feature_sizes
        self.conditional_size = encoder_layer_sizes[0]

        # category-related map
        # category-related map
        self.map_enc = []
        for i, size in enumerate(feature_sizes):
            self.map_enc.append(nn.Linear(size, encoder_layer_sizes[0]))

        # category-related map
        self.map_dec = []
        for i, size in enumerate(feature_sizes):
            self.map_dec.append(nn.Linear(decoder_layer_sizes[-1], size))

        self.encoder = Encoder(encoder_layer_sizes, latent_size, category_index)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, category_index, self.conditional_size)


    def forward(self, x, c, category):

        x = self.map_enc[category[0]](x)
        c = self.map_enc[category[1]](c)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)
        recon_x = self.map_dec[category[0]](recon_x)
        recon_x = F.sigmoid(recon_x)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c, category):

        c = self.map_enc[category[1]](c)
        recon_x = self.decoder(z, c)
        recon_x = self.map_dec[category[0]](recon_x)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, category_index):
        super().__init__()

        self.category_index = category_index
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

    def __init__(self, layer_sizes, latent_size, category_index, conditional_size):

        super().__init__()

        self.MLP = nn.Sequential()
        self.category_index = category_index
        input_size = latent_size + conditional_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c):

        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x