# infer.py
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.encoder import Encoder
from models.decoder import Decoder

def load_npy_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    data = []
    for f in files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # [1, H, W]
        data.append((f, arr))
    return data

def infer(cfg, stage3_ckpt_path, sim_folder=None, real_folder=None, save_dir="inference_outputs", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------
    # Load Stage3 models
    # ----------------------
    latent_dim = cfg['train']['latent_dim']
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    ckpt = torch.load(stage3_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print(f"Loaded Stage3 checkpoint: {stage3_ckpt_path}")

    # ----------------------
    # 推理模拟数据
    # ----------------------
    if sim_folder is not None:
        sim_out_dir = os.path.join(save_dir, "sim")
        os.makedirs(sim_out_dir, exist_ok=True)
        sim_data = load_npy_files(sim_folder)
        print(f"Found {len(sim_data)} simulated samples")

        for fpath, x_sim in sim_data:
            filename = os.path.splitext(os.path.basename(fpath))[0]
            x_tensor = torch.from_numpy(x_sim).unsqueeze(0).to(device)  # [1, 1, H, W]

            with torch.no_grad():
                mu, logvar = encoder(x_tensor)
                z = mu  # 推理不采样
                phi_hat = decoder(z)
            phi_hat_cpu = phi_hat.squeeze().cpu().numpy()

            # 保存 npy
            np.save(os.path.join(sim_out_dir, f"{filename}_phi_hat.npy"), phi_hat_cpu)
            # 保存图像
            plt.imshow(phi_hat_cpu, cmap='gray')
            plt.colorbar()
            plt.savefig(os.path.join(sim_out_dir, f"{filename}_phi_hat.png"))
            plt.close()

            # 如果有 ground truth 同时在 sim_folder 也可以加载 phi_sim 做指标
            phi_sim_path = fpath.replace("WRAP", "TRUE")
            if os.path.exists(phi_sim_path):
                phi_sim = np.load(phi_sim_path)
                l1 = np.mean(np.abs(phi_hat_cpu - phi_sim))
                l2 = np.mean((phi_hat_cpu - phi_sim)**2)
                psnr_val = psnr(phi_sim, phi_hat_cpu)
                ssim_val = ssim(phi_sim, phi_hat_cpu)
                print(f"[SIM] {filename} -> L1:{l1:.4f}, L2:{l2:.4f}, PSNR:{psnr_val:.2f}, SSIM:{ssim_val:.4f}")

    # ----------------------
    # 推理真实数据
    # ----------------------
    if real_folder is not None:
        real_out_dir = os.path.join(save_dir, "real")
        os.makedirs(real_out_dir, exist_ok=True)
        real_data = load_npy_files(real_folder)
        print(f"Found {len(real_data)} real samples")

        for fpath, x_real in real_data:
            filename = os.path.splitext(os.path.basename(fpath))[0]
            x_tensor = torch.from_numpy(x_real).unsqueeze(0).to(device)  # [1, 1, H, W]

            with torch.no_grad():
                mu, logvar = encoder(x_tensor)
                z = mu
                phi_hat = decoder(z)
            phi_hat_cpu = phi_hat.squeeze().cpu().numpy()

            # 保存 npy
            np.save(os.path.join(real_out_dir, f"{filename}_phi_hat.npy"), phi_hat_cpu)
            # 保存图像
            plt.imshow(phi_hat_cpu, cmap='gray')
            plt.colorbar()
            plt.savefig(os.path.join(real_out_dir, f"{filename}_phi_hat.png"))
            plt.close()
            print(f"[REAL] {filename} -> output saved")

if __name__ == "__main__":
    import yaml

    config_path = "config/pcvae_gan.yaml"
    stage3_ckpt_path = "checkpoints/stage3/checkpoint_epoch10.pth"

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 模拟数据路径
    sim_folder = "/DATA/TEST/WRAP"
    # 真实数据路径
    real_folder = "/DATA/TEST/REAL"

    infer(cfg, stage3_ckpt_path, sim_folder=sim_folder, real_folder=real_folder, save_dir="inference_outputs", device="cuda")
