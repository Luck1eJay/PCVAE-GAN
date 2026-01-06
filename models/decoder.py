import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, channels=[64,32,1], z_dim=32):
        super().__init__()

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
        h = self.fc(z).view(-1, 64, 8, 8)
        phi = self.deconv(h)
        return phi
