import torch
import torch.nn.functional as F

def wrap_loss(phi_hat, x_real, wrap_fn):
    """
    Physics / wrap consistency loss
    phi_hat: [batch, H, W] decoder output
    x_real: [batch, H, W] real wrapped phase
    wrap_fn: function W(phi) -> [-pi, pi)
    """
    return F.l1_loss(wrap_fn(phi_hat), x_real)
