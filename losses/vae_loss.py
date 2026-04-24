import torch
import torch.nn.functional as F
def _kl_gaussian(mu, logvar, prior_mu=None, prior_logvar=None):
    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)
    var = torch.exp(logvar)
    prior_var = torch.exp(prior_logvar)
    kl = 0.5 * (prior_logvar - logvar + (var + (mu - prior_mu).pow(2)) / prior_var - 1.0)
    return kl.mean()
def kl_loss(mu, logvar, prior_mu=None, prior_logvar=None, free_bits=0.0, kl_weights=None, return_levels=False):
    if isinstance(mu, (list, tuple)):
        if not isinstance(logvar, (list, tuple)):
            raise TypeError('logvar must be a list/tuple when mu is a list/tuple')
        level_count = len(mu)
        weights = list(kl_weights) if kl_weights is not None else [1.0] * level_count
        if len(weights) < level_count:
            weights = weights + [1.0] * (level_count - len(weights))
        total = torch.zeros((), device=mu[0].device, dtype=mu[0].dtype)
        levels = []
        for idx, (mu_i, logvar_i) in enumerate(zip(mu, logvar)):
            prior_mu_i = None if prior_mu is None else prior_mu[idx]
            prior_logvar_i = None if prior_logvar is None else prior_logvar[idx]
            level_kl = _kl_gaussian(mu_i, logvar_i, prior_mu_i, prior_logvar_i)
            if free_bits and free_bits > 0.0:
                level_kl = torch.clamp(level_kl, min=float(free_bits))
            weighted = weights[idx] * level_kl
            total = total + weighted
            levels.append(weighted)
        if return_levels:
            return total, levels
        return total
    level_kl = _kl_gaussian(mu, logvar, prior_mu, prior_logvar)
    if free_bits and free_bits > 0.0:
        level_kl = torch.clamp(level_kl, min=float(free_bits))
    if return_levels:
        return level_kl, [level_kl]
    return level_kl
def vae_loss(
    phi_hat,
    phi,
    mu,
    logvar,
    lambda_geo=1.0,
    lambda_kl=1e-4,
    prior_mu=None,
    prior_logvar=None,
    free_bits=0.0,
    kl_weights=None,
    vq_loss=None,
    lambda_vq=1.0,
    return_kl_levels=False,
):
    loss_geo = F.l1_loss(phi_hat, phi)
    loss_kl, kl_levels = kl_loss(
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        free_bits=free_bits,
        kl_weights=kl_weights,
        return_levels=True,
    )
    total = lambda_geo * loss_geo + lambda_kl * loss_kl
    if vq_loss is not None:
        total = total + float(lambda_vq) * vq_loss
    if return_kl_levels:
        return total, loss_geo, loss_kl, kl_levels
    return total, loss_geo, loss_kl
