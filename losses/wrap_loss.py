import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

def wrap_loss(phi_pred, x_real, wrap_func):
    """
    物理一致性损失：保证解缠后的相位经过缠绕算子后与真实观测相位一致
    """
    phi_wrapped = wrap_func(phi_pred)
    loss = mse_loss(phi_wrapped, x_real)
    return loss

