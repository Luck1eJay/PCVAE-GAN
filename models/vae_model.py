from models.encoder import Encoder
from models.decoder import Decoder
from utils.utils import reparameterize as _reparameterize
import torch.nn as nn

class VAEModel(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, x_wrap):
        mu, logvar = self.encoder(x_wrap)
        z = _reparameterize(mu, logvar)
        phi_hat = self.decoder(x_wrap, z)
        return phi_hat, mu, logvar, z

    def reparameterize(self, mu, logvar):
        """
        暴露一个实例方法 reparameterize，使外部可以通过 vae.reparameterize 调用。
        这样可以兼容仓库中其他地方（如 inference.py）对 vae.reparameterize 的调用。
        """
        return _reparameterize(mu, logvar)