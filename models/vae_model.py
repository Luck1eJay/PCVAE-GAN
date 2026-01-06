from models.encoder import Encoder
from models.decoder import Decoder
from utils.utils import reparameterize
import torch.nn as nn

class VAEModel(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, x_wrap):
        mu, logvar = self.encoder(x_wrap)
        z = reparameterize(mu, logvar)
        phi_hat = self.decoder(x_wrap, z)
        return phi_hat, mu, logvar, z
