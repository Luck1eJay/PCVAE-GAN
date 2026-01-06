import torch.nn as nn

class LatentDiscriminator(nn.Module):
    """
    判别器用于 latent space 对齐
    """
    def __init__(self, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, z):
        return self.net(z)
