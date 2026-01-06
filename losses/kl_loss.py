import torch

def kl_loss(mu, logvar):
    """
    KL divergence between q(z|x) and N(0, I)
    """
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
