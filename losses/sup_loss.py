import torch.nn.functional as F

def sup_loss(phi_pred, phi_gt):
    """
    Supervised MSE loss for simulated data
    """
    return F.mse_loss(phi_pred, phi_gt)
