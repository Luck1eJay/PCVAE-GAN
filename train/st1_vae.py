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

def train_stage1(cfg, checkpoint_dir, device='cuda', n_samples=5):
    """
    Stage1: VAE training (simulated data)
    n_samples: 每个输入生成的多解数量 K
    """
    train_cfg = cfg['train']
    data_cfg = cfg['data']
    loss_cfg = cfg['loss']

    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr_encoder']
    num_epochs = train_cfg['num_epochs_stage1']
    latent_dim = cfg['model']['latent_dim']

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
    train_dataset = PhaseDataset(wrap_dir=data_cfg['sim_data'], phi_dir=data_cfg['sim_phi'], mode='sim')
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

            # ---------------------------
            # Encoder
            # ---------------------------
            mu, logvar = encoder(x_sim)
            batch_size, latent_dim = mu.size()

            # ---------------------------
            # 同一个 x 采样 n_samples 个 z
            # ---------------------------
            eps = torch.randn(batch_size, n_samples, latent_dim, device=device)
            z = mu.unsqueeze(1) + eps * torch.exp(0.5 * logvar).unsqueeze(1)  # [B, K, D]

            # ---------------------------
            # Decoder forward
            # ---------------------------
            phi_hat = decoder(z)  # [B, K, 1, H, W]

            # ---------------------------
            # 监督损失: best-of-K L1
            # ---------------------------
            # phi_sim: [B, 1, H, W] -> 扩展到 [B, K, H, W] 方便计算
            phi_sim_expand = phi_sim.unsqueeze(1).expand(-1, n_samples, -1, -1)
            l1_all = torch.abs(phi_hat.squeeze(2) - phi_sim_expand)  # [B, K, H, W]
            loss_geo = l1_all.view(batch_size, n_samples, -1).mean(dim=2).min(dim=1)[0].mean()  # best-of-K

            # ---------------------------
            # KL loss
            # ---------------------------
            loss_kl = kl_loss(mu, logvar)

            # ---------------------------
            # diversity loss
            # ---------------------------
            loss_div = diversity_loss(phi_hat.squeeze(2))  # [B, K, H, W]

            # ---------------------------
            # 总损失
            # ---------------------------
            loss_total = loss_geo + lambda_kl * loss_kl + lambda_div * loss_div

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss_total.backward()
            opt_enc.step()
            opt_dec.step()

            total_loss_epoch += loss_total.item()

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"[Stage1] Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # ---------------------------
        # checkpoint
        # ---------------------------
        if epoch % 10 == 0:
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

        # ---------------------------
        # 可视化最后一个 batch 的多解
        # ---------------------------
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # 只取 batch 的前 4 个样本，每个样本显示 K 个解
            n_plot = min(4, x_sim.size(0))
            for i in range(n_plot):
                plt.figure(figsize=(3*n_samples, 3))
                for k in range(n_samples):
                    plt.subplot(1, n_samples, k+1)
                    plt.title(f'sample {k}')
                    plt.imshow(phi_hat[i, k, 0].cpu(), cmap='gray')
                    plt.axis('off')
                plt.show()
        encoder.train()
        decoder.train()

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
