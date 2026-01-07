import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base_channels*8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*8, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.conv(x)
        h = self.adaptive_pool(h)
        h = h.view(batch_size, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar, n_samples=1):
        """Reparameterization trick with multiple samples"""
        batch_size, latent_dim = mu.size()
        eps = torch.randn(batch_size, n_samples, latent_dim, device=mu.device)
        z = mu.unsqueeze(1) + eps * torch.exp(0.5 * logvar).unsqueeze(1)
        # z: [batch, n_samples, latent_dim]
        return z

