import pdb
import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Dict, Tuple


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

    def encoder(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder_layers(x)
        return self.fc_mean(x), self.fc_var(x)

    def reparameterize(self, mu: Tensor, std: Tensor) -> Tensor:
        out = []
        for _ in range(self.latents_to_sample):
            eps = torch.randn_like(std)
            out.append(eps * std + mu)
        out = torch.stack(out)

        return out.mean(dim=0)

    def decoder(self, z: Tensor) -> Tensor:
        z = self.decoder_layers(z)
        z = z.view(-1, 64, 7, 7)
        return self.conv_transpose(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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

    def encoder(self, x: Tensor) -> Tensor:
        for layer in self.encoder_layers:
            x = torch.relu(layer(x))
        return self.fc_mean(x), self.fc_var(x)

    def reparameterize(self, mu: Tensor, std: Tensor) -> Tensor:
        out = []
        for _ in range(self.latents_to_sample):
            eps = torch.randn_like(std)
            out.append(eps * std + mu)
        out = torch.stack(out)

        return out.mean(dim=0)

    def decoder(self, z: Tensor) -> Tensor:
        for layer in self.decoder_layers:
            z = torch.relu(layer(z))
        return torch.sigmoid(self.fc_final(z))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        return self.decoder(z), mean, var


# Credit to https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py for this implementation of DDPM
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def block(in_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 7, padding=3),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )


class DummyEpsModel(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            block(n_channel, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(
        self, n_sample: int, size, device, return_trajectory=False
    ) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        trajectory = []

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            if return_trajectory:
                trajectory.append(x_i)

        if return_trajectory:
            return x_i, trajectory

        return x_i
