import torch
from models.vae_model import VAEModel
from utils.metrics import load_cfg, reparameterize
import numpy as np

# 加载配置信息
cfg = load_cfg("config/pcvae_gan.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化并移动到 device
vae = VAEModel(z_dim=cfg['model']['z_dim']).to(device)

# 加载 checkpoint：使用 map_location 确保在 CPU-only 环境下也能加载；
# 同时兼容 checkpoint 为 dict (包含 'state_dict') 或直接 state_dict 的情况
_ckpt = torch.load("checkpoints/vae_stage3.pth", map_location=device)
_state = _ckpt.get('state_dict', _ckpt) if isinstance(_ckpt, dict) else _ckpt
vae.load_state_dict(_state)
vae.eval()

def infer_single(x_wrap):
    """
    接受一个 wrap 图像张量（torch.Tensor），返回 phi 的预测（CPU 张量）。
    注意：确保传入的是形状 (B, C, H, W) 的 torch.Tensor。
    """
    x = x_wrap.to(device)
    with torch.no_grad():
        mu, logvar = vae.encoder(x)
        # 使用 mu 作为确定性编码输出（不采样）
        phi = vae.decoder(x, mu)
    return phi.cpu()

def infer_multi(x_wrap, N=5):
    """
    基于重参数化采样多次隐变量，返回一个长度为 N 的 list，元素为 CPU 张量。
    """
    x = x_wrap.to(device)
    with torch.no_grad():
        mu, logvar = vae.encoder(x)
        phis = []
        for _ in range(N):
            z = reparameterize(mu, logvar).to(device)
            phis.append(vae.decoder(x, z).cpu())
    return phis
