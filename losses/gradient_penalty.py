import torch
from torch.autograd import grad

def gradient_penalty(D, z_real, z_sim, device='cuda'):
    """
    R1-style gradient penalty for latent GAN
    """
    alpha = torch.rand(z_real.size(0), 1).to(device)
    alpha = alpha.expand_as(z_real)

    interpolated = alpha * z_real + (1 - alpha) * z_sim
    interpolated.requires_grad_(True)

    D_interpolated = D(interpolated)

    gradients = grad(
        outputs=D_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(D_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

