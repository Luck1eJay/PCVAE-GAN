import torch

def latent_gan_loss(z_sim, z_real, D):
    """
    R3GAN / Relativistic latent GAN loss
    z_sim: latent from simulated data
    z_real: latent from real data
    D: latent discriminator
    """
    # Discriminator loss
    D_real = D(z_real)
    D_sim = D(z_sim.detach())
    loss_D = -torch.mean(torch.log(torch.sigmoid(D_sim - D_real)))

    # Encoder as generator loss
    loss_G = -torch.mean(torch.log(torch.sigmoid(D_real - D_sim)))

    return loss_D, loss_G

def gradient_penalty(D, z_real, z_sim, lambda_gp=10.0):
    """
    R1 / R2 gradient penalty for latent discriminator
    """
    z_hat = 0.5 * (z_real + z_sim.detach())
    z_hat.requires_grad_(True)
    d_hat = D(z_hat)
    grad = torch.autograd.grad(outputs=d_hat.sum(), inputs=z_hat,
                               create_graph=True, retain_graph=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp
