import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    VAE Encoder
    输入：wrap 图像 x_wrap (1,H,W)
    输出：latent 分布参数 mu, logvar
    """
    def __init__(self, channels=[1,32,64], z_dim=32):
        super().__init__()
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(channels[-1]*8*8, z_dim)  # 假设 H=W=32
        self.fc_logvar = nn.Linear(channels[-1]*8*8, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
