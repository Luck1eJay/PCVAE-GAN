import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    VAE Encoder
    输入：wrap 图像 x_wrap (B, C, H, W)
    输出：latent 分布参数 mu, logvar
    """
    def __init__(self, channels=None, z_dim=32):
        super().__init__()
        if channels is None:
            channels = [1, 32, 64]
        self._channels = channels

        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

        # 使用全局自适应池化避免对输入 HxW 做硬编码假设
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(channels[-1] * 1 * 1, z_dim)
        self.fc_logvar = nn.Linear(channels[-1] * 1 * 1, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.global_pool(h)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
