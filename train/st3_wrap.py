import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.decoder import Decoder
from data.dataset import PhaseDataset
from losses.wrap_loss import wrap_loss
from losses.gradient_loss import gradient_loss

def train_stage3(cfg, stage2_ckpt_dir, checkpoint_dir, device='cuda'):
    """
    Stage3: Real data physical consistency (wrap + gradient)
    """
    train_cfg = cfg['train']
    data_cfg = cfg['data']
    loss_cfg = cfg['loss']

    batch_size = train_cfg['batch_size']
    lr_enc = train_cfg['lr_encoder']
    lr_dec = train_cfg['lr_decoder']
    num_epochs = train_cfg['num_epochs_stage3']
    latent_dim = train_cfg['model']['latent_dim']

    lambda_wrap = loss_cfg.get('lambda_wrap', 1.0)
    lambda_grad = loss_cfg.get('lambda_grad', 1.0)

    # ---------------------------
    # 模型
    # ---------------------------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    opt_enc = optim.Adam(encoder.parameters(), lr=lr_enc, betas=(0.5, 0.999))
    opt_dec = optim.Adam(decoder.parameters(), lr=lr_dec, betas=(0.5, 0.999))

    # ---------------------------
    # 加载 Stage2 checkpoint
    # ---------------------------
    stage2_ckpt_path = os.path.join(stage2_ckpt_dir, f"checkpoint_epoch{train_cfg.get('stage2_load_epoch',10)}.pth")
    ckpt = torch.load(stage2_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    print(f"Loaded Stage2 encoder checkpoint from {stage2_ckpt_path}")

    decoder.eval()  # 冻结 decoder
    for p in decoder.parameters():
        p.requires_grad = False

    # ---------------------------
    # 数据
    # ---------------------------
    train_dataset = PhaseDataset(sim_data=None, real_data=data_cfg['real_data'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # Stage3 Training
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.eval()
        total_loss = 0.0

        for x_real in train_loader:
            x_real = x_real.to(device)

            mu_real, logvar_real = encoder(x_real)
            z_real = mu_real + torch.randn_like(mu_real) * torch.exp(0.5 * logvar_real)
            phi_hat = decoder(z_real)

            loss_wrap = wrap_loss(phi_hat, x_real)
            loss_grad = gradient_loss(phi_hat, phi_hat.detach())
            loss_total = lambda_wrap * loss_wrap + lambda_grad * loss_grad

            opt_enc.zero_grad()
            loss_total.backward()
            opt_enc.step()

            total_loss += loss_total.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % cfg['logging'].get('print_interval',1) == 0:
            print(f"[Stage3] Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # checkpoint
        if epoch % cfg['logging'].get('save_interval',10) == 0:
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
