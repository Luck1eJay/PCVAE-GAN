import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

def geometric_loss(phi_pred, phi_gt):
    """
    模拟数据上的几何监督损失
    L_geo = || phi_pred - phi_gt ||_2^2
    """
    return mse_loss(phi_pred, phi_gt)

def kl_divergence(mu, logvar):
    """
    KL 散度约束隐变量分布
    L_KL = D_KL(q(z|x_sim) || N(0,I))
    """
    batch_size = mu.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl / batch_size

def vae_loss(phi_pred, phi_gt, mu, logvar, kl_weight=1e-3):
    """
    总 VAE loss
    L_VAE = L_geo + kl_weight * L_KL
    返回 (总 loss, L_geo, L_KL)
    """
    L_geo = geometric_loss(phi_pred, phi_gt)
    L_kl = kl_divergence(mu, logvar) * kl_weight
    return L_geo + L_kl, L_geo, L_kl
