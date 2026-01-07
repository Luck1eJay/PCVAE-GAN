import torch
import torch.nn as nn

mse_loss = nn.MSELoss()


def wrap_loss(phi_pred, x_real, wrap_func):
    """
    物理一致性损失：保证解缠后的相位经过缠绕算子后与真实观测相位一致
    L_wrap = || W(phi_pred) - x_real ||_2^2

    参数：
    - phi_pred: 解码器输出的解缠相位，形状 [B, C, H, W] 或 [B, H, W]
    - x_real: 真实缠绕相位，形状与 phi_pred 对应
    - wrap_func: 缠绕算子 W(phi)，函数形式，比如 lambda phi: (phi + pi) % (2*pi) - pi

    返回：
    - 平均 batch 的 L2 损失
    """
    # 将预测相位经过缠绕算子
    phi_wrapped = wrap_func(phi_pred)

    # 计算均方误差
    loss = mse_loss(phi_wrapped, x_real)
    return loss
