import torch

def wrap_phase(phi):
    return torch.atan2(torch.sin(phi), torch.cos(phi))

def wrap_loss(phi_pred, x_wrap):
    """
    Enforce wrap consistency:
    wrap(phi_pred) ≈ x_wrap
    """
    phi_wrapped = wrap_phase(phi_pred)
    return torch.mean(torch.abs(phi_wrapped - x_wrap))
