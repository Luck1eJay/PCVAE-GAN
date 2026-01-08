import torch
import torch.nn.functional as F
import math

def rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))

def psnr(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def ssim(pred, gt, C1=0.01**2, C2=0.03**2):
    """
    简化版 SSIM（单通道，全图）
    无 skimage 依赖，论文可用
    """
    mu_x = pred.mean()
    mu_y = gt.mean()

    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return ssim_val
