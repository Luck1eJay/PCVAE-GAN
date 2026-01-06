import torch
from models.vae_model import VAEModel
from utils.utils import load_cfg
import numpy as np

cfg = load_cfg("config/pcvae_gan.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = VAEModel(z_dim=cfg['model']['z_dim']).to(device)
vae.load_state_dict(torch.load("checkpoints/vae_stage3.pth"))
vae.eval()

def infer_single(x_wrap):
    with torch.no_grad():
        mu, logvar = vae.encoder(x_wrap)
        phi = vae.decoder(x_wrap, mu)
    return phi

def infer_multi(x_wrap, N=5):
    with torch.no_grad():
        mu, logvar = vae.encoder(x_wrap)
        phis = []
        for _ in range(N):
            z = vae.reparameterize(mu, logvar)
            phis.append(vae.decoder(x_wrap, z))
    return phis

