import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
    def forward(self, x):
        return nn.ReLU(inplace=True)(x + self.block(x))

class NVAEEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32, latent_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, stride=2, padding=1), # 256→128
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            ResBlock(base_channels)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1), # 128→64
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True),
            ResBlock(base_channels*2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1), # 64→32
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True),
            ResBlock(base_channels*4)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1), # 32→16
            nn.GroupNorm(8, base_channels*8),
            nn.ReLU(inplace=True),
            ResBlock(base_channels*8)
        )
        self.mu_logvar1 = nn.Conv2d(base_channels, latent_channels*2, 1)
        self.mu_logvar2 = nn.Conv2d(base_channels*2, latent_channels*2, 1)
        self.mu_logvar3 = nn.Conv2d(base_channels*4, latent_channels*2, 1)
        self.mu_logvar4 = nn.Conv2d(base_channels*8, latent_channels*2, 1)

    def forward(self, x):
        features = []
        x1 = self.enc1(x)  # [B, C, 128, 128]
        features.append(x1)
        x2 = self.enc2(x1) # [B, 2C, 64, 64]
        features.append(x2)
        x3 = self.enc3(x2) # [B, 4C, 32, 32]
        features.append(x3)
        x4 = self.enc4(x3) # [B, 8C, 16, 16]
        features.append(x4)

        mus, logvars = [], []
        for feat, conv in zip(features, [self.mu_logvar1, self.mu_logvar2, self.mu_logvar3, self.mu_logvar4]):
            mu_logvar = conv(feat)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def sample(self, mus, logvars):
        zs = []
        for mu, logvar in zip(mus, logvars):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            zs.append(mu + eps * std)
        return zs  # [z1: 128x128, z2: 64x64, z3:32x32, z4: 16x16]

