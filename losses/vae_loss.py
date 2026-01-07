import torch
import torch.nn.functional as F

def kl_loss(mu, logvar):
    """
    KL Divergence between latent distribution and N(0,I)
    mu: [batch, latent_dim]
    logvar: [batch, latent_dim]
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def geo_loss(phi_hat, phi_sim):
    """
    L1 Reconstruction loss on simulated data
    phi_hat: [batch, H, W]
    phi_sim: [batch, H, W]
    """
    return F.l1_loss(phi_hat, phi_sim)

def vae_loss(phi_hat, phi_sim, mu, logvar, lambda_geo=1.0, lambda_kl=0.01):
    """
    Combined VAE loss: weighted sum of geo + KL
    """
    loss_geo = geo_loss(phi_hat, phi_sim)
    loss_kl = kl_loss(mu, logvar)
    return lambda_geo * loss_geo + lambda_kl * loss_kl
