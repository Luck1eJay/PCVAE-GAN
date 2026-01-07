import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, base_channels=32, output_channels=1):
        """
        latent_dim: latent space 维度
        base_channels: decoder 基础卷积通道数
        output_channels: 输出相位图通道数，一般为1
        """
        super().__init__()

        # 将 latent vector 扩展成 feature map
        self.fc = nn.Linear(latent_dim, base_channels*8*4*4)  # 假设最低分辨率 4x4

        # 转置卷积上采样
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, output_channels, 3, stride=1, padding=1)
        )

    def forward(self, z):
        """
        z: [batch, latent_dim] 或 [batch, n_samples, latent_dim]
        返回: [batch, 1, H, W] 或 [batch, n_samples, 1, H, W]
        """
        is_multi = False
        if z.dim() == 3:
            # batch × n_samples × latent_dim
            is_multi = True
            batch, n_samples, latent_dim = z.size()
            z = z.view(batch*n_samples, latent_dim)

        h = self.fc(z).unsqueeze(-1).unsqueeze(-1)  # [batch*n_samples, channels, 1, 1] → 4x4
        out = self.upconv(h)

        if is_multi:
            batch = batch
            out = out.view(batch, n_samples, *out.shape[1:])
        return out


