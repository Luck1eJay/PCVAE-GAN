import torch
from torch.utils.data import DataLoader
from models.vae_model import VAEModel
from models.discriminator import LatentDiscriminator
from data.sim_dataset import SimDataset
from data.real_dataset import RealDataset
from losses.latent_gan_loss import gan_loss_D, gan_loss_G
from utils.utils import load_cfg, save_model
import os

# ------------------------------
# 1️ 配置 & 设备
# ------------------------------
cfg = load_cfg("config/pcvae_gan.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 2️ 数据
# ------------------------------
sim_dataset = SimDataset(cfg['data']['sim_path'])
real_dataset = RealDataset(cfg['data']['real_path'])

sim_loader = DataLoader(sim_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
real_loader = DataLoader(real_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

# ------------------------------
# 3️ 模型
# ------------------------------
vae = VAEModel(z_dim=cfg['model']['z_dim']).to(device)
disc = LatentDiscriminator(z_dim=cfg['model']['z_dim']).to(device)

# 加载 Stage1 预训练权重
vae.load_state_dict(torch.load("checkpoints/vae_stage1.pth"))

# 冻结 Decoder（非常重要）
for p in vae.decoder.parameters():
    p.requires_grad = False

# ------------------------------
# 4️ 优化器
# ------------------------------
opt_E = torch.optim.Adam(vae.encoder.parameters(), lr=cfg['train']['lr_vae'])
opt_D = torch.optim.Adam(disc.parameters(), lr=cfg['train']['lr_disc'])

# ------------------------------
# 5️ 训练
# ------------------------------
for epoch in range(cfg['train']['epochs_stage2']):
    for (x_sim, _), x_real in zip(sim_loader, real_loader):

        x_sim = x_sim.to(device)
        x_real = x_real.to(device)

        # ==========================
        # 5.1 训练判别器 D
        # ==========================
        with torch.no_grad():
            _, _, _, z_sim = vae(x_sim)
            _, _, _, z_real = vae(x_real)

        pred_sim = disc(z_sim.detach())
        pred_real = disc(z_real.detach())

        loss_D = gan_loss_D(pred_sim, pred_real)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ==========================
        # 5.2 训练 Encoder（GAN-G）
        # ==========================
        _, _, _, z_real = vae(x_real)
        pred_real = disc(z_real)

        loss_G = gan_loss_G(pred_real)

        opt_E.zero_grad()
        loss_G.backward()
        opt_E.step()

    print(f"[Stage2][Epoch {epoch+1}] D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")

# ------------------------------
# 6️ 保存模型
# ------------------------------
os.makedirs("checkpoints", exist_ok=True)
save_model(vae, "checkpoints/vae_stage2.pth")
save_model(disc, "checkpoints/disc_stage2.pth")
