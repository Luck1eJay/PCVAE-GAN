import torch
import torch.nn.functional as F

def gan_loss_D(pred_sim, pred_real):
    """
    Discriminator loss
    """
    loss_sim = F.binary_cross_entropy_with_logits(
        pred_sim, torch.ones_like(pred_sim)
    )
    loss_real = F.binary_cross_entropy_with_logits(
        pred_real, torch.zeros_like(pred_real)
    )
    return loss_sim + loss_real

def gan_loss_G(pred_real):
    """
    Encoder (Generator) loss
    """
    return F.binary_cross_entropy_with_logits(
        pred_real, torch.ones_like(pred_real)
    )
