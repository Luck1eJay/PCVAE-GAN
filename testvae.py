import os
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.vaedata import UnwrappedPhaseDataset
from models.vae import NVAE  # 或 SpatialVAE，根据你的实现


# 确保你的NVAE结构参数和训练阶段对应

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_compare_plot(phi_gt, phi_hat, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    vmin = min(phi_gt.min().item(), phi_hat.min().item())
    vmax = max(phi_gt.max().item(), phi_hat.max().item())
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(phi_gt, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.title("GT (unwrapped)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(phi_hat, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.title("Reconstructed")
    plt.colorbar()
    plt.suptitle(f"Test Sample {idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_test_{idx}.png"))
    plt.close()


@torch.no_grad()
def test():
    cfg = load_config("config/pcvae_gan.yaml")
    device = torch.device(
        cfg["test"]["device"] if "test" in cfg and "device" in cfg["test"] else cfg["train"]["device"])

    # 1. 加载/实例化模型
    model = NVAE(
        input_channels=1,
        base_channels=32,  # 如果确定你的网络就是32
        latent_channels=cfg["model"]["latent_channels"]
    ).to(device)

    # 2. 加载权重
    weight_path = cfg["test"].get("model_path", None) \
        if ("test" in cfg and "model_path" in cfg["test"]) \
        else os.path.join(cfg["train"]["checkpoint_dir"], "spatial_vae_epoch_197.pth")  # 或最后一轮模型
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 3. 加载数据
    dataset = UnwrappedPhaseDataset(
        phi_dir=cfg["data"]["test_phi"],  # 测试集数据目录
        phi_key=cfg["data"].get("phi_key", None),
        transform=None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    save_dir = os.path.join(cfg["test"].get("out_dir", "test_out"))
    os.makedirs(save_dir, exist_ok=True)

    total_l1 = 0.0
    sample_num = 0

    for idx, phi in enumerate(dataloader):
        phi = phi.to(device)  # [1, 1, H, W]

        # NVAE和单层VAE的接口兼容
        phi_hat, mu, logvar = model(phi)  # [1, 1, H, W] 或 [1, C, H, W]
        # 如果网络输出是[1, 1, H, W]
        pred = phi_hat[0, 0].detach().cpu().numpy()
        gt = phi[0, 0].detach().cpu().numpy()

        # 保存对比图
        save_compare_plot(gt, pred, idx, save_dir)

        # 计算L1重建误差
        total_l1 += abs(pred - gt).mean()
        sample_num += 1

        if idx < 5:
            print(f"Sample {idx}: L1 error = {abs(pred - gt).mean():.6f}")

    print("=" * 20)
    print(f"Mean L1 error (all test): {total_l1 / sample_num:.6f}")
    print(f"重建结果图像保存在: {save_dir}/")
    print("=" * 20)


if __name__ == "__main__":
    test()