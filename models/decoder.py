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

import torch
import torch.nn as nn

class NVAEDecoder(nn.Module):
    def __init__(self, latent_channels, base_channels, output_channels):
        super().__init__()
        # 假设 base=32, latent=64
        # 对应Encoder的4层输出feature：256,128,64,32

        self.block4 = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels*8, 3, padding=1), # 64→256
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1) # 256→128,16→32
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels*4 + latent_channels, base_channels*4, 3, padding=1), # 128+64→128
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1) # 128→64,32→64
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels*2 + latent_channels, base_channels*2, 3, padding=1), # 64+64→64
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1) # 64→32,64→128
        self.block1 = nn.Sequential(
            nn.Conv2d(base_channels + latent_channels, base_channels, 3, padding=1), # 32+64→32
            nn.ReLU(inplace=True),
        )
        self.final_up = nn.ConvTranspose2d(base_channels, output_channels, 4, stride=2, padding=1) # 32→1

    def forward(self, zs):
        z1, z2, z3, z4 = zs
        x = self.block4(z4)  # (B, 256, 16, 16)
        x = self.up3(x)      # (B, 128, 32, 32)
        x = torch.cat([x, z3], dim=1) # (B, 192, 32, 32)
        x = self.block3(x)   # (B, 128, 32, 32)
        x = self.up2(x)      # (B, 64, 64, 64)
        x = torch.cat([x, z2], dim=1) # (B, 128, 64, 64)
        x = self.block2(x)   # (B, 64, 64, 64)
        x = self.up1(x)      # (B, 32, 128, 128)
        x = torch.cat([x, z1], dim=1) # (B, 96, 128, 128)
        x = self.block1(x)   # (B, 32, 128, 128)
        x = self.final_up(x) # (B, 1, 256, 256)
        return x