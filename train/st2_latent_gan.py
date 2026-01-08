import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_discriminator import LatentDiscriminator
from data.dataset import PhaseDataset
from losses.latent_gan_loss import latent_gan_loss, gradient_penalty  # R3GAN + GP

def train_stage2(cfg, stage1_ckpt_dir, checkpoint_dir, device='cuda'):
    """
    Stage2: Latent space alignment via R3GAN
    """
    train_cfg = cfg['train']
    data_cfg = cfg['data']
    loss_cfg = cfg['loss']

    batch_size = train_cfg['batch_size']
    lr_enc = train_cfg['lr_encoder']
    lr_disc = train_cfg['lr_discriminator']
    num_epochs = train_cfg['num_epochs_stage2']
    latent_dim = train_cfg['latent_dim']
    lambda_gp = loss_cfg.get('lambda_gp', 10.0)

    # ---------------------------
    # 初始化模型
    # ---------------------------
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    discriminator = LatentDiscriminator(latent_dim=latent_dim).to(device)

    opt_enc = optim.Adam(encoder.parameters(), lr=lr_enc, betas=(0.5,0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5,0.999))

    # ---------------------------
    # Stage1 checkpoint
    # ---------------------------
    stage1_ckpt_path = os.path.join(stage1_ckpt_dir, 'checkpoint_epoch10.pth')
    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    print(f"Loaded Stage1 checkpoint from {stage1_ckpt_path}")

    # ---------------------------
    # 数据
    # ---------------------------
    train_dataset = PhaseDataset(sim_data=data_cfg['sim_data'], real_data=data_cfg['real_data'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # Stage2 Training
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        discriminator.train()
        total_loss_enc, total_loss_disc = 0.0, 0.0

        for x_sim, x_real in train_loader:
            x_sim, x_real = x_sim.to(device), x_real.to(device)

            # Encoder forward
            mu_sim, logvar_sim = encoder(x_sim)
            z_sim = mu_sim + torch.randn_like(mu_sim) * torch.exp(0.5*logvar_sim)

            mu_real, logvar_real = encoder(x_real)
            z_real = mu_real + torch.randn_like(mu_real) * torch.exp(0.5*logvar_real)

            # -----------------
            # R3GAN + gradient penalty
            # -----------------
            loss_d, loss_g = latent_gan_loss(z_sim, z_real, discriminator)
            gp = gradient_penalty(discriminator, z_real, z_sim, lambda_gp=lambda_gp)
            total_loss_d = loss_d + gp
            total_loss_g = loss_g

            # 更新 discriminator
            opt_disc.zero_grad()
            total_loss_d.backward(retain_graph=True)
            opt_disc.step()

            # 更新 encoder
            opt_enc.zero_grad()
            total_loss_g.backward()
            opt_enc.step()

            total_loss_enc += total_loss_g.item()
            total_loss_disc += total_loss_d.item()

        print(f"[Stage2] Epoch {epoch}/{num_epochs}, Avg Enc Loss: {total_loss_enc/len(train_loader):.4f}, Avg Disc Loss: {total_loss_disc/len(train_loader):.4f}")

        # -----------------
        # checkpoint
        # -----------------
        if epoch % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch
            }, os.path.join(checkpoint_dir, f'stage2_epoch{epoch}.pth'))
            print(f"Checkpoint saved: {checkpoint_dir}/stage2_epoch{epoch}.pth")

    print("Stage2 training finished.")
    return encoder, discriminator

