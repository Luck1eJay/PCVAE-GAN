import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from losses.vae_loss import kl_loss, geo_loss
# from losses.diversity_loss import diversity_loss
from losses.gradient_loss import gradient_loss
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
    lr_encoder = train_cfg['lr_encoder']
    lr_decoder = train_cfg['lr_decoder']

    num_epochs = train_cfg['num_epochs_stage1']
    latent_dim = cfg['model']['latent_dim']

    lambda_kl = loss_cfg.get('beta', 1.0)
    # lambda_div = loss_cfg.get('lambda_div', 0.1)
    lambda_grad = loss_cfg.get('lambda_grad', 1.0)
    # ---------------------------
    # 模型
    # ---------------------------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    opt_enc = optim.Adam(encoder.parameters(), lr=lr_encoder)
    opt_dec = optim.Adam(decoder.parameters(), lr=lr_decoder)

    # ---------------------------
    # 数据
    # ---------------------------
    train_dataset = PhaseDataset(
        wrap_dir=data_cfg['sim_data'],
        phi_dir=data_cfg['sim_phi'],
        mode='sim',
        wrap_key=data_cfg.get('wrap_key', 'input'),
        phi_key=data_cfg.get('phi_key', 'output'),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # 训练
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss_epoch = 0.0

        last_x_sim = None
        last_phi_sim = None
        last_phi_hat = None

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

            last_x_sim = x_sim
            last_phi_sim = phi_sim
            last_phi_hat = phi_hat

            # ---------------------------
            # 监督损失: best-of-K L1
            # ---------------------------
            # phi_sim: [B, 1, H, W] -> 扩展到 [B, K, 1, H, W] 方便计算
            phi_sim_expand = phi_sim.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
            l1_all = torch.abs(phi_hat.squeeze(2) - phi_sim_expand.squeeze(2))  # [B, K, H, W]
            l1_mean = l1_all.view(batch_size, n_samples, -1).mean(dim=2)
            loss_geo = l1_mean.min(dim=1)[0].mean()  # best-of-K

            best_idx = l1_mean.min(dim=1)[1]
            phi_hat_best = phi_hat[torch.arange(batch_size, device=device), best_idx, :, :, :]  # [B, 1, H, W]
            loss_grad = gradient_loss(phi_hat_best, phi_sim)

            # ---------------------------
            # KL loss
            # ---------------------------
            loss_kl = kl_loss(mu, logvar)

            # ---------------------------
            # diversity loss
            # ---------------------------
            # loss_div = diversity_loss(phi_hat.squeeze(2))  # [B, K, H, W]


            print("x_sim range:", x_sim.min().item(), x_sim.max().item())
            print("phi_sim range:", phi_sim.min().item(), phi_sim.max().item())
            print(
                "loss_geo:", loss_geo.item(),
                "loss_kl:", loss_kl.item(),
                # "loss_div:", loss_div.item(),
                "loss_grad:", loss_grad.item()
            )
            # ---------------------------
            # 总损失
            # ---------------------------
            # loss_total = loss_geo + lambda_kl * loss_kl + lambda_div * loss_div + lambda_grad * loss_grad
            loss_total = loss_geo + lambda_kl * loss_kl  + lambda_grad * loss_grad

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
        # 可视化：wrapped + labels + 多解输出（每张子图独立 colorbar）
        # ---------------------------
        if last_phi_hat is not None:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                idx = last_phi_hat.size(0) - 1  # 最后一张
                wrapped = last_x_sim[idx, 0].cpu()
                label = last_phi_sim[idx, 0].cpu()
                outputs = last_phi_hat[idx, :, 0].cpu()  # [K, H, W]

                # labels 与 outputs 使用同一范围（真实相位范围）
                vmin_label = label.min().item()
                vmax_label = label.max().item()

                n_cols = 2 + n_samples
                fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

                # wrapped
                axes[0].set_title("wrapped")
                im0 = axes[0].imshow(wrapped, cmap='hsv', vmin=-3.1416, vmax=3.1416)
                axes[0].axis('off')
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                # label
                axes[1].set_title("label")
                im1 = axes[1].imshow(label, cmap='hsv', vmin=vmin_label, vmax=vmax_label)
                axes[1].axis('off')
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # outputs
                for k in range(n_samples):
                    axes[2 + k].set_title(f'pred {k}')
                    imk = axes[2 + k].imshow(outputs[k], cmap='hsv', vmin=vmin_label, vmax=vmax_label)
                    axes[2 + k].axis('off')
                    fig.colorbar(imk, ax=axes[2 + k], fraction=0.046, pad=0.04)

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
    config_path = 'config/pcvae_gan.yaml'
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
    device_str = cfg.train.get('device', 'cuda:0')
    if torch.cuda.is_available():
        device = torch.device(device_str)
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # ---------------------------
    # 4. 启动 Stage1 训练
    # ---------------------------
    train_stage1(cfg, checkpoint_base, device=device)