import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# ---------------------------
# 导入模型和 loss
# ---------------------------
from models.encoder import Encoder
from models.decoder import Decoder
from losses.vae_loss import vae_loss  # 封装后的 VAE loss
from losses.diversity_loss import diversity_loss
from data.dataset import PhaseDataset

# ---------------------------
# 读取配置
# ---------------------------
config_path = './config/pcvae_gan.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

train_cfg = cfg['train']
data_cfg = cfg['data']
loss_cfg = cfg['loss']
model_name = cfg.get('model_name', 'PCVAE-GAN')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = train_cfg['batch_size']
lr = train_cfg['lr']
num_epochs = train_cfg['num_epochs_stage1']
latent_dim = train_cfg['latent_dim']
n_samples = train_cfg.get('n_samples', 1)

# 自动创建 checkpoint 文件夹
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], model_name, 'stage1')
os.makedirs(checkpoint_dir, exist_ok=True)

lambda_geo = loss_cfg.get('lambda_geo', 1.0)
lambda_kl = loss_cfg.get('lambda_kl', 0.01)
lambda_div = loss_cfg.get('lambda_div', 0.1)

# ---------------------------
# 初始化模型
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
# Stage1: VAE + Diversity Training
# ---------------------------
for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    total_loss_epoch = 0.0

    for x_sim, phi_sim in train_loader:
        x_sim, phi_sim = x_sim.to(device), phi_sim.to(device)

        # -----------------
        # Forward pass
        # -----------------
        mu, logvar = encoder(x_sim)
        # 采样 latent z, 支持多解
        eps = torch.randn(mu.size(0), latent_dim, device=device)
        z = mu + eps * torch.exp(0.5 * logvar)
        phi_hat = decoder(z)

        # -----------------
        # Compute losses
        # -----------------
        loss_vae = vae_loss(phi_hat, phi_sim, mu, logvar, lambda_geo, lambda_kl)
        loss_div = diversity_loss(phi_hat)

        total_loss = loss_vae + lambda_div * loss_div

        # -----------------
        # Backprop
        # -----------------
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        total_loss.backward()
        opt_enc.step()
        opt_dec.step()

        total_loss_epoch += total_loss.item()

    avg_loss = total_loss_epoch / len(train_loader)
    print(f"[Stage1] Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # -----------------
    # Save checkpoint every 10 epochs
    # -----------------
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'opt_enc_state_dict': opt_enc.state_dict(),
            'opt_dec_state_dict': opt_dec.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

print("Stage1 training finished.")
