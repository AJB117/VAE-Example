import torch
import torch.nn as nn

from torch import Tensor
from typing import List


class VAE_CNN(nn.Module):
    def __init__(self, latent_dim: int, latents_to_sample: int = 1):
        super(VAE_CNN, self).__init__()

        self.latent_dim = latent_dim
        self.latents_to_sample = latents_to_sample

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64 * 7 * 7), nn.ReLU()
        )

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encoder(self, x: Tensor):
        x = self.encoder_layers(x)
        return self.fc_mean(x), self.fc_var(x)

    def reparameterize(self, mu: Tensor, std: Tensor):
        out = []
        for _ in range(self.latents_to_sample):
            eps = torch.randn_like(std)
            out.append(eps * std + mu)
        out = torch.stack(out)

        return out.mean(dim=0)

    def decoder(self, z: Tensor):
        z = self.decoder_layers(z)
        z = z.view(-1, 64, 7, 7)
        return self.conv_transpose(z)

    def forward(self, x: Tensor):
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var)
        return self.decoder(z), mu, var


class VAE_MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layer_sizes: List[int],
        latent_dim: int,
        latents_to_sample: int = 1,
    ):
        super(VAE_MLP, self).__init__()
        self.in_dim = in_dim
        self.latents_to_sample = latents_to_sample

        self.fc_mean = nn.Linear(hidden_layer_sizes[-1], latent_dim)  # mu
        self.fc_var = nn.Linear(hidden_layer_sizes[-1], latent_dim)  # sigma

        self.encoder_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            if i == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_layer_sizes[i]

            self.encoder_layers.append(nn.Linear(in_dim, hidden_layer_sizes[i + 1]))

        self.decoder_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1, 0, -1):
            if i == len(hidden_layer_sizes) - 1:
                in_dim = latent_dim
            else:
                in_dim = hidden_layer_sizes[i]

            self.decoder_layers.append(nn.Linear(in_dim, hidden_layer_sizes[i - 1]))

        self.fc_final = nn.Linear(hidden_layer_sizes[0], self.in_dim)

    def encoder(self, x: Tensor):
        for layer in self.encoder_layers:
            x = torch.relu(layer(x))
        return self.fc_mean(x), self.fc_var(x)

    def reparameterize(self, mu: Tensor, std: Tensor):
        out = []
        for _ in range(self.latents_to_sample):
            eps = torch.randn_like(std)
            out.append(eps * std + mu)
        out = torch.stack(out)

        return out.mean(dim=0)

    def decoder(self, z: Tensor):
        for layer in self.decoder_layers:
            z = torch.relu(layer(z))
        return torch.sigmoid(self.fc_final(z))

    def forward(self, x: Tensor):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        return self.decoder(z), mean, var
