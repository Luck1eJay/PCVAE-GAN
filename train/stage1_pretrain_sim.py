import torch
from torch.utils.data import DataLoader
from models.vae_model import VAEModel
from data.sim_dataset import SimDataset
from losses.sup_loss import sup_loss
from losses.wrap_loss import wrap_loss
from losses.kl_loss import kl_loss
from utils.utils import load_cfg, save_model
import os

# ------------------------------
# 1 配置和设备初始化
# ------------------------------
cfg = load_cfg("config/pcvae_gan.yaml")   #
# 读取 YAML 配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 2️ 数据加载
# ------------------------------
sim_dataset = SimDataset(cfg['data']['sim_path'])
sim_loader = DataLoader(sim_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

# ------------------------------
# 3️ 模型 + 优化器
# ------------------------------
vae = VAEModel(z_dim=cfg['model']['z_dim']).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=cfg['train']['lr_vae'])

# ------------------------------
# 4️ 训练循环
# ------------------------------
for epoch in range(cfg['train']['epochs_stage1']):
    total_loss = 0
    for x_wrap, phi_gt in sim_loader:
        # 1️ 将数据送入设备
        x_wrap, phi_gt = x_wrap.to(device), phi_gt.to(device)

        # 2️ 梯度清零
        optimizer.zero_grad()

        # 3️
        # forward：Encoder -> latent -> Decoder
        phi_hat, mu, logvar, z = vae(x_wrap)

        # ------------------------------
        # 4️ 损失计算
        # ------------------------------
        # 4.1 监督损失（模拟数据）
        L_sup = sup_loss(phi_hat, phi_gt)
        # 数学公式：
        # L_sup = mean( (phi_hat - phi_gt)^2 )

        # 4.2 wrap consistency loss（模型解出的 phi wrap 回去应该接近原输入 wrap）
        L_wrap = wrap_loss(phi_hat, x_wrap)
        # L_wrap = mean(| wrap(phi_hat) - x_wrap |)
        # wrap(phi) = atan2(sin(phi), cos(phi))

        # 4.3 KL loss（VAE latent 正则）
        L_kl = kl_loss(mu, logvar)
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        # 4.4 总损失 = 监督 + wrap + KL
        loss = (
            cfg['loss']['sup_weight'] * L_sup +
            cfg['loss']['wrap_weight'] * L_wrap +
            cfg['train']['kl_weight'] * L_kl
        )

        # 5️ 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ------------------------------
    # 6️ 日志打印
    # ------------------------------
    avg_loss = total_loss / len(sim_loader)
    print(f"Epoch [{epoch+1}/{cfg['train']['epochs_stage1']}], Avg Loss: {avg_loss:.4f}")

    # 7️ 每 N epoch 保存模型
    if (epoch+1) % 10 == 0:
        save_path = os.path.join("checkpoints", f"vae_stage1_epoch{epoch+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)
        save_model(vae, save_path)
        print(f"Saved model checkpoint: {save_path}")
