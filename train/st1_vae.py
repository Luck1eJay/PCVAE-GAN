import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.decoder import Decoder
from losses.vae_loss import kl_loss, geo_loss
from losses.diversity_loss import diversity_loss
from data.dataset import PhaseDataset

def train_stage1(cfg, checkpoint_dir, device='cuda'):
    """Stage1: VAE training (simulated data)"""
    train_cfg = cfg['train']
    data_cfg = cfg['data']
    loss_cfg = cfg['loss']

    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs_stage1']
    latent_dim = train_cfg['model']['latent_dim']

    lambda_kl = loss_cfg.get('beta', 1.0)
    lambda_div = loss_cfg.get('lambda_div', 0.1)

    # ---------------------------
    # 模型
    # ---------------------------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    opt_enc = optim.Adam(encoder.parameters(), lr=lr)
    opt_dec = optim.Adam(decoder.parameters(), lr=lr)

    # ---------------------------
    # 数据
    # ---------------------------
    train_dataset = PhaseDataset(sim_data=data_cfg['sim_data'], real_data=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # 训练
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss_epoch = 0.0

        for x_sim, phi_sim in train_loader:
            x_sim = x_sim.to(device)
            phi_sim = phi_sim.to(device)

            mu, logvar = encoder(x_sim)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            phi_hat = decoder(z)

            loss_geo = geo_loss(phi_hat, phi_sim)
            loss_kl = kl_loss(mu, logvar)
            # loss_div = diversity_loss(phi_hat)

            # loss_total = loss_geo + lambda_kl * loss_kl + lambda_div * loss_div
            loss_total = loss_geo + lambda_kl * loss_kl

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss_total.backward()
            opt_enc.step()
            opt_dec.step()

            total_loss_epoch += loss_total.item()

        avg_loss = total_loss_epoch / len(train_loader)
        if epoch % cfg['logging'].get('print_interval', 1) == 0:
            print(f"[Stage1] Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # checkpoint
        if epoch % cfg['logging'].get('save_interval', 10) == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'opt_enc_state_dict': opt_enc.state_dict(),
                'opt_dec_state_dict': opt_dec.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    return encoder, decoder
