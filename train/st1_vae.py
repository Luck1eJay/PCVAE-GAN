import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from losses.vae_loss import kl_loss, geo_loss
from losses.diversity_loss import diversity_loss
from data.dataset import PhaseDataset

def train_stage1(cfg, checkpoint_dir, device='cuda'):
    """Stage1: VAE training (simulated data)"""

    train_cfg = cfg.train
    data_cfg  = cfg.data
    loss_cfg  = cfg.loss

    batch_size = train_cfg.batch_size
    num_epochs = train_cfg.num_epochs_stage1
    latent_dim = cfg.model.latent_dim

    lr_enc = train_cfg.lr_encoder
    lr_dec = train_cfg.lr_decoder

    lambda_kl = loss_cfg.beta
    lambda_div =loss_cfg.lambda_div
    # ---------------------------
    # 模型
    # ---------------------------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    opt_enc = optim.Adam(encoder.parameters(), lr=lr_enc)
    opt_dec = optim.Adam(decoder.parameters(), lr=lr_dec)

    # ---------------------------
    # 数据
    # ---------------------------
    train_dataset = PhaseDataset(
        wrap_dir=data_cfg.sim_data,
        phi_dir=data_cfg.sim_phi,
        mode='sim'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    last_batch = None

    # ---------------------------
    # 训练
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss_epoch = 0.0

        for x_sim, phi_sim in train_loader:
            x_sim  = x_sim.to(device)
            phi_sim = phi_sim.to(device)
            last_batch = (x_sim, phi_sim)

            mu, logvar = encoder(x_sim)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            phi_hat = decoder(z)

            loss_geo = geo_loss(phi_hat, phi_sim)
            loss_kl  = kl_loss(mu, logvar)
            loss_total = loss_geo + lambda_kl * loss_kl

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss_total.backward()
            opt_enc.step()
            opt_dec.step()

            total_loss_epoch += loss_total.item()

        print(f"[Stage1] Epoch {epoch}/{num_epochs}, Avg Loss: {total_loss_epoch / len(train_loader):.4f}")

        # checkpoint
        if epoch % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': epoch
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))

    # ---------------------------
    # 可视化最后一个 batch
    # ---------------------------
    encoder.eval()
    decoder.eval()

    x_sim, phi_sim = last_batch
    with torch.no_grad():
        mu, logvar = encoder(x_sim)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        phi_hat = decoder(z)

    n_plot = min(4, x_sim.size(0))
    for i in range(n_plot):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Input wrapped phase')
        plt.imshow(x_sim[i, 0].cpu(), cmap='gray')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title('Reconstructed continuous phase')
        plt.imshow(phi_hat[i, 0].cpu(), cmap='gray')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    return encoder, decoder

if __name__ == '__main__':
    import os
    import torch
    from omegaconf import OmegaConf
    # ---------------------------
    # 1. 加载配置
    # ---------------------------
    config_path = '/home/junjie/PCVAE-GAN/config/pcvae_gan.yaml'
    cfg = OmegaConf.load(config_path)
    print("Config loaded:")
    print(cfg)

    # ---------------------------
    # 2. 设置 checkpoint 目录
    # ---------------------------
    checkpoint_base = cfg.train.checkpoint_dir
    os.makedirs(checkpoint_base, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_base}")

    # ---------------------------
    # 3. 指定 GPU
    # ---------------------------
    gpu_id = 1  # 可修改为 0/1/2
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id} for training")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # ---------------------------
    # 4. 启动 Stage1 训练
    # ---------------------------
    train_stage1(cfg, checkpoint_base, device=device)
