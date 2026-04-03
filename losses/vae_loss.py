# vae_loss.py
import torch.nn.functional as F

def kl_loss(mu, logvar):
    """
    KL散度损失
    支持：mu, logvar 为单层 (tensor) 或多层 (list of tensor)
    Shape: [B, C, H, W] 或 [B, D] (单层)；多层则为列表
    """
    if isinstance(mu, list):  # 多层KL
        return sum([
            -0.5 * (1 + lv - mu_.pow(2) - lv.exp()).mean()
            for mu_, lv in zip(mu, logvar)
        ])
    else:  # 单层KL
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

def vae_loss(phi_hat, phi, mu, logvar, lambda_geo=1.0, lambda_kl=1e-4):
    """
    综合VAE损失：重建 + KL
    """
    loss_geo = F.l1_loss(phi_hat, phi)
    loss_kl = kl_loss(mu, logvar)
    return lambda_geo * loss_geo + lambda_kl * loss_kl, loss_geo, loss_kl
