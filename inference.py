import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder


# -----------------------------
# Utils
# -----------------------------
def l1_error(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()

def rmse(pred, gt):
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()

def save_image(arr, path, cmap="jet"):
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -----------------------------
# Dataset
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, wrap_dir, true_dir=None):
        self.wrap_dir = wrap_dir
        self.true_dir = true_dir
        self.files = sorted(os.listdir(wrap_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        wrap = np.load(os.path.join(self.wrap_dir, name)).astype(np.float32)
        wrap = torch.from_numpy(wrap).unsqueeze(0)  # [1,H,W]

        if self.true_dir is not None:
            true = np.load(os.path.join(self.true_dir, name)).astype(np.float32)
            true = torch.from_numpy(true).unsqueeze(0)
            return name, wrap, true
        else:
            return name, wrap


# -----------------------------
# Inference Core
# -----------------------------
def run_inference(cfg):
    device = torch.device(cfg["inference"]["device"])

    # Model
    encoder = Encoder(latent_dim=cfg["model"]["latent_dim"]).to(device)
    decoder = Decoder(latent_dim=cfg["model"]["latent_dim"]).to(device)

    ckpt = torch.load(cfg["model"]["checkpoint"], map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    K = cfg["inference"]["num_samples"]
    save_root = cfg["inference"]["save_dir"]
    os.makedirs(save_root, exist_ok=True)

    # -------------------------
    # 1. Simulated Data
    # -------------------------
    if "sim_wrap" in cfg["data"]:
        print("Running inference on simulated data...")
        sim_dir = os.path.join(save_root, "sim")
        os.makedirs(sim_dir, exist_ok=True)

        dataset = TestDataset(cfg["data"]["sim_wrap"], cfg["data"]["sim_true"])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        metrics = []

        with torch.no_grad():
            for name, wrap, gt in loader:
                name = name[0].replace(".npy", "")
                wrap = wrap.to(device)
                gt = gt.to(device)

                mu, logvar = encoder(wrap)
                std = torch.exp(0.5 * logvar)

                samples = []
                for k in range(K):
                    eps = torch.randn_like(mu)
                    z = mu + eps * std
                    phi = decoder(z)
                    samples.append(phi)

                samples = torch.stack(samples, dim=0)  # [K,1,H,W]
                mean = samples.mean(dim=0)
                std_map = samples.std(dim=0)

                # Metrics
                l1 = l1_error(mean, gt)
                r = rmse(mean, gt)
                metrics.append(f"{name}: L1={l1:.4f}, RMSE={r:.4f}")

                # Save npy
                np.save(f"{sim_dir}/{name}_mean.npy", mean.cpu().numpy())
                np.save(f"{sim_dir}/{name}_std.npy", std_map.cpu().numpy())
                for k in range(K):
                    np.save(f"{sim_dir}/{name}_k{k}.npy", samples[k].cpu().numpy())

                # Save images
                save_image(mean[0, 0].cpu().numpy(), f"{sim_dir}/{name}_mean.png")
                save_image(std_map[0, 0].cpu().numpy(), f"{sim_dir}/{name}_std.png")

        with open(f"{sim_dir}/metrics.txt", "w") as f:
            f.write("\n".join(metrics))

    # -------------------------
    # 2. Real Data
    # -------------------------
    if "real_wrap" in cfg["data"]:
        print("Running inference on real data...")
        real_dir = os.path.join(save_root, "real")
        os.makedirs(real_dir, exist_ok=True)

        dataset = TestDataset(cfg["data"]["real_wrap"])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for name, wrap in loader:
                name = name[0].replace(".npy", "")
                wrap = wrap.to(device)

                mu, logvar = encoder(wrap)
                std = torch.exp(0.5 * logvar)

                samples = []
                for k in range(K):
                    eps = torch.randn_like(mu)
                    z = mu + eps * std
                    phi = decoder(z)
                    samples.append(phi)

                samples = torch.stack(samples, dim=0)
                mean = samples.mean(dim=0)
                std_map = samples.std(dim=0)

                np.save(f"{real_dir}/{name}_mean.npy", mean.cpu().numpy())
                np.save(f"{real_dir}/{name}_std.npy", std_map.cpu().numpy())
                for k in range(K):
                    np.save(f"{real_dir}/{name}_k{k}.npy", samples[k].cpu().numpy())

                save_image(mean[0, 0].cpu().numpy(), f"{real_dir}/{name}_mean.png")
                save_image(std_map[0, 0].cpu().numpy(), f"{real_dir}/{name}_std.png")


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/infer.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)
