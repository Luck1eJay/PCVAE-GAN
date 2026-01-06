import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, channels=None, z_dim=32):
        super().__init__()
        # 避免使用可变默认参数
        if channels is None:
            channels = [64, 32, 1]

        self._channels = channels
        self.fc = nn.Linear(z_dim, channels[0] * 8 * 8)

        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            if i < len(channels) - 2:
                layers.append(nn.ReLU())

        self.deconv = nn.Sequential(*layers)

    def forward(self, x_wrap, z):
        """
        保留 x_wrap 参数以兼容现有调用签名，但当前实现未使用 x_wrap。
        将 fc 输出 reshape 时使用 channels[0] 而非硬编码 64。
        """
        c0 = self._channels[0]
        h = self.fc(z).view(-1, c0, 8, 8)
        phi = self.deconv(h)
        return phi
