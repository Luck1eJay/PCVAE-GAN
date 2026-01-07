import torch
import torch.nn as nn
import torch.autograd as autograd


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)  # 输出标量，用于 R3GAN loss
        )

    def forward(self, z):
        return self.net(z)


def gradient_penalty(discriminator, real_latent, fake_latent, device='cuda'):
    """
    WGAN-GP 风格的梯度惩罚，用于稳定 R3GAN 训练
    """
    alpha = torch.rand(real_latent.size(0), 1, device=device)
    alpha = alpha.expand_as(real_latent)
    interpolates = alpha * real_latent + (1 - alpha) * fake_latent
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
