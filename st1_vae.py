import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from losses.vae_loss import kl_loss
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
    # lambda_geo = loss_cfg.get('lambda_geo', 1.0)

    # Allow n_samples to be set from config (overrides function arg)
    n_samples = int(train_cfg.get('n_samples_stage1', n_samples))
    # recon_scale: 用来把 L1 映射到 log p(x|z) 的尺度（IWAE 权重敏感项）
    recon_scale = loss_cfg.get('recon_scale', 1.0)
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
        total_iwae_epoch = 0.0
        total_kl_epoch = 0.0
        total_grad_epoch = 0.0
        total_recon_epoch = 0.0  # mean recon_l1 averaged per batch
        total_ess_epoch = 0.0
        total_wentropy_epoch = 0.0
        total_perpixelstd_epoch = 0.0
        batch_count = 0

        last_x_sim = None
        last_phi_sim = None
        last_phi_hat = None

        for batch_idx, (x_sim, phi_sim) in enumerate(train_loader):
            batch_count += 1
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

            # # ---------------------------
            # # 监督损失: best-of-K L1
            # # ---------------------------
            # # phi_sim: [B, 1, H, W] -> 扩展到 [B, K, 1, H, W] 方便计算
            # phi_sim_expand = phi_sim.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
            # l1_all = torch.abs(phi_hat.squeeze(2) - phi_sim_expand.squeeze(2))  # [B, K, H, W]
            # l1_mean = l1_all.view(batch_size, n_samples, -1).mean(dim=2)
            # loss_geo = l1_mean.min(dim=1)[0].mean()  # best-of-K
            #
            # best_idx = l1_mean.min(dim=1)[1]
            # phi_hat_best = phi_hat[torch.arange(batch_size, device=device), best_idx, :, :, :]  # [B, 1, H, W]
            # loss_grad = gradient_loss(phi_hat_best, phi_sim)

            # ---------------------------
            # 监督损失: 用 IWAE 替代 best-of-K，并用 L1 构造 log p(x|z)
            # ---------------------------
            # phi_sim: [B, 1, H, W] -> 扩展到 [B, K, 1, H, W] 方便计算
            K = n_samples
            phi_hat_squeezed = phi_hat.squeeze(2)  # [B, K, H, W]
            phi_sim_expand = phi_sim.unsqueeze(1).expand(-1, K, -1, -1, -1)  # [B, K, H, W]
            phi_sim_expand_squeezed = phi_sim_expand.squeeze(2)  # [B, K, H, W]

            # Now compute per-pixel absolute diffs exactly like your working snippet
            l1_all = torch.abs(phi_hat_squeezed - phi_sim_expand_squeezed)  # [B, K, H, W]
            l1_mean = l1_all.view(batch_size, K, -1).mean(dim=2)  # [B, K]  per-sample mean L1
            recon_l1 = l1_mean  # [B, K]

            # approximate log p(x|z) via negative scaled L1
            log_px_z = - recon_scale * recon_l1  # [B, K]

            # log p(z) under standard normal prior (ignore constants)
            z_flat = z.view(batch_size * K, latent_dim)
            log_pz = (-0.5 * (z_flat ** 2).sum(dim=1)).view(batch_size, K)  # [B, K]

            # log q(z|x) for Gaussian q = N(mu, sigma^2)
            mu_exp = mu.unsqueeze(1).expand(-1, K, -1).contiguous().view(batch_size * K, latent_dim)
            logvar_exp = logvar.unsqueeze(1).expand(-1, K, -1).contiguous().view(batch_size * K, latent_dim)
            var_exp = torch.exp(logvar_exp)
            log_qz_x = (-0.5 * (((z_flat - mu_exp) ** 2) / var_exp + logvar_exp).sum(dim=1)).view(batch_size, K)

            # importance log weights
            log_w = log_px_z + log_pz - log_qz_x  # [B, K]

            # stable IWAE estimate: log(1/K * sum_k w_k)
            max_log_w, _ = log_w.max(dim=1, keepdim=True)  # [B,1]
            log_mean_w = (max_log_w.squeeze(1) + torch.log(torch.exp(log_w - max_log_w).mean(dim=1)))  # [B]
            loss_iwae = - log_mean_w.mean()  # scalar, negative IWAE lower bound

            # normalized importance weights
            weights = torch.softmax(log_w, dim=1)  # [B, K]

            # weighted reconstruction for gradient loss
            w_view = weights.view(batch_size, K, 1, 1)
            phi_hat_weighted = (w_view * phi_hat_squeezed).sum(dim=1, keepdim=True)  # [B,1,H,W]

            # gradient consistency loss on weighted reconstruction
            loss_grad = gradient_loss(phi_hat_weighted, phi_sim)
            # ---------------------------
            # KL loss
            # ---------------------------
            loss_kl = kl_loss(mu, logvar)

            # ---------------------------
            # diversity loss
            # ---------------------------

            # loss_div = diversity_loss(phi_hat.squeeze(2))  # [B, K, H, W]

            # ---------------------------
            # accumulate per-batch diagnostics for epoch-level summary
            # ---------------------------
            # recon_mean: scalar (mean L1 over B,K)
            recon_mean_batch = float(recon_l1.mean().item())

            with torch.no_grad():
                # ESS per sample then mean across batch
                ess = 1.0 / (weights.pow(2).sum(dim=1) + 1e-12)  # [B]
                ess_mean_batch = float(ess.mean().item())
                # weight entropy per sample then mean
                w_entropy_batch = float((-(weights * (weights + 1e-12).log()).sum(dim=1)).mean().item())
                # per-pixel std across K averaged over batch (diversity)
                per_pixel_std_batch = float(phi_hat_squeezed.std(dim=1).view(batch_size, -1).mean().item())

            total_iwae_epoch += float(loss_iwae.item())
            total_kl_epoch += float(loss_kl.item())
            total_grad_epoch += float(loss_grad.item())
            total_recon_epoch += recon_mean_batch
            total_ess_epoch += ess_mean_batch
            total_wentropy_epoch += w_entropy_batch
            total_perpixelstd_epoch += per_pixel_std_batch

            # ---------------------------
            # 总损失
            # ---------------------------
            # loss_total = loss_geo + lambda_kl * loss_kl + lambda_div * loss_div + lambda_grad * loss_grad
            # loss_total = lambda_geo * loss_geo + lambda_kl * loss_kl  + lambda_grad * loss_grad
            loss_total = loss_iwae + lambda_kl * loss_kl  + lambda_grad * loss_grad

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss_total.backward()
            opt_enc.step()
            opt_dec.step()

            total_loss_epoch += loss_total.item()

            # end of epoch: compute averages for each component
        if batch_count > 0:
            avg_loss = total_loss_epoch / batch_count
            avg_iwae = total_iwae_epoch / batch_count
            avg_kl = total_kl_epoch / batch_count
            avg_grad = total_grad_epoch / batch_count
            avg_recon = total_recon_epoch / batch_count
            avg_ess = total_ess_epoch / batch_count
            avg_wentropy = total_wentropy_epoch / batch_count
            avg_perpixelstd = total_perpixelstd_epoch / batch_count
        else:
            avg_loss = avg_iwae = avg_kl = avg_grad = avg_recon = avg_ess = avg_wentropy = avg_perpixelstd = 0.0

        print(f"[Stage1] Epoch {epoch}/{num_epochs}, Avg Total Loss: {avg_loss:.4f}")
        print(f"  Component averages: loss_iwae={avg_iwae:.4f}, loss_kl={avg_kl:.4f}, loss_grad={avg_grad:.4f}")
        print(f"  Recon L1 mean={avg_recon:.6f}, per-pixel-std={avg_perpixelstd:.6f}")
        print(f"  Weights: ESS_mean={avg_ess:.2f} (<=K={n_samples}), entropy_mean={avg_wentropy:.4f}")

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
        # 可视化：wrapped + labels + 最优解
        # ---------------------------
        if last_phi_hat is not None:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                # 选择 batch 中最后一个样本用于可视化（和原实现一致）
                idx = last_phi_hat.size(0) - 1
                wrapped = last_x_sim[idx, 0].cpu()  # [H, W]
                label = last_phi_sim[idx, 0].cpu()  # [H, W]
                outputs = last_phi_hat[idx, :, 0].cpu()  # [K, H, W]

                # 计算每个预测与 label 的平均 L1（按像素平均）
                # outputs: [K, H, W], label: [H, W] -> broadcast
                diffs = (outputs - label.unsqueeze(0)).abs()  # [K, H, W]
                l1_per_k = diffs.view(diffs.size(0), -1).mean(dim=1)  # [K]

                # 找到与 label 最近的预测索引
                best_k = int(torch.argmin(l1_per_k).item())
                best_output = outputs[best_k]  # [H, W]
                best_l1 = float(l1_per_k[best_k].item())

                # 差分图（signed difference）：best_output - label
                diff = best_output - label  # signed diff (can be negative)
                # 差分色阶对称化：以绝对最大值为范围（便于正负偏差比较）
                diff_abs_max = max(abs(float(diff.min().item())), abs(float(diff.max().item())))
                if diff_abs_max == 0:
                    diff_vmin, diff_vmax = -1e-6, 1e-6
                else:
                    diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

                # 确保 label 与 best_output 共享同一色阶（避免视觉误导）
                combined_min = float(min(label.min().item(), best_output.min().item()))
                combined_max = float(max(label.max().item(), best_output.max().item()))
                # 若范围退化（几乎恒定），稍微扩展以保证 colorbar 可用
                if abs(combined_max - combined_min) < 1e-6:
                    combined_min -= 1e-3
                    combined_max += 1e-3

                # 打印诊断信息
                print(
                    f"[Vis] best_k={best_k}, plain_L1={best_l1:.6f}, label_range=({label.min().item():.4f},{label.max().item():.4f}), output_range=({best_output.min().item():.4f},{best_output.max().item():.4f})")

                # 绘图：四列：wrapped / label / best_pred / diff
                n_cols = 4
                fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

                # 1) wrapped（若是相位通常在 [-pi,pi]，仍用该固定范围；若非相位仍按原值显示）
                axes[0].set_title("wrapped")
                try:
                    im0 = axes[0].imshow(wrapped, cmap='rainbow', vmin=-3.1416, vmax=3.1416)
                except Exception:
                    im0 = axes[0].imshow(wrapped, cmap='rainbow')
                axes[0].axis('off')
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                # 2) label（与 best_pred 共享色阶）
                axes[1].set_title("label")
                im1 = axes[1].imshow(label, cmap='rainbow', vmin=combined_min, vmax=combined_max)
                axes[1].axis('off')
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # 3) best prediction（同一色阶）
                axes[2].set_title(f'best pred (k={best_k})')
                im2 = axes[2].imshow(best_output, cmap='rainbow', vmin=combined_min, vmax=combined_max)
                axes[2].axis('off')
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                # 4) signed difference map（对称色阶）
                axes[3].set_title('diff (pred - label)')
                im3 = axes[3].imshow(diff, cmap='rainbow', vmin=diff_vmin, vmax=diff_vmax)
                axes[3].axis('off')
                fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

                plt.tight_layout()
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