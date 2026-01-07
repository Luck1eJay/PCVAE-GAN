import torch
from torch.utils.data import DataLoader
from models.vae_model import VAEModel
from data.real_dataset import RealDataset
from losses.wrap_loss import wrap_loss
from losses.diversity_loss import diversity_loss
from losses.kl_loss import kl_loss
from utils.utils import load_cfg, save_model

# ------------------------------
# 1️ 配置 & 设备
# ------------------------------
cfg = load_cfg("config/pcvae_gan.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 2️ 数据（只用真实 wrap）
# ------------------------------
real_dataset = RealDataset(cfg['data']['real_path'])
real_loader = DataLoader(real_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

# ------------------------------
# 3️ 模型
# ------------------------------
vae = VAEModel(z_dim=cfg['model']['z_dim']).to(device)
vae.load_state_dict(torch.load("checkpoints/vae_stage2.pth"))

optimizer = torch.optim.Adam(vae.parameters(), lr=cfg['train']['lr_vae'])

# ------------------------------
# 4️ 多解采样数
# ------------------------------
N_samples = 4   # 每个输入采 4 个 latent

# ------------------------------
# 5️ 训练
# ------------------------------
for epoch in range(cfg['train']['epochs_stage3']):
    for x_wrap in real_loader:
        x_wrap = x_wrap.to(device)

        phi_list = []
        mu, logvar = vae.encoder(x_wrap)

        # 5.1 多次采样
        for _ in range(N_samples):
            z = vae.reparameterize(mu, logvar)
            phi = vae.decoder(x_wrap, z)
            phi_list.append(phi)

        # ------------------------------
        # 5.2 损失
        # ------------------------------
        # wrap consistency（所有解都必须合理）
        L_wrap = sum([wrap_loss(phi, x_wrap) for phi in phi_list]) / N_samples

        # 多解多样性（鼓励解之间不同）
        L_div = diversity_loss(phi_list)

        # KL 正则
        L_kl = kl_loss(mu, logvar)

        loss = (
            cfg['loss']['wrap_weight'] * L_wrap +
            cfg['train']['kl_weight'] * L_kl +
            cfg['train']['diversity_weight'] * L_div
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Stage3][Epoch {epoch+1}] Loss={loss.item():.4f}")

# ------------------------------
# 6️ 保存模型
# ------------------------------
save_model(vae, "checkpoints/vae_stage3.pth")
