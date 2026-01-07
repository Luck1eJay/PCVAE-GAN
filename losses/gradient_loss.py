import torch
import torch.nn.functional as F

def gradient_loss(phi_hat, phi_target):
    """
    Preserve local phase jump / high frequency structure
    """
    def gradient(x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dx, dy
    dx_hat, dy_hat = gradient(phi_hat)
    dx_target, dy_target = gradient(phi_target)
    return F.l1_loss(dx_hat, dx_target) + F.l1_loss(dy_hat, dy_target)
