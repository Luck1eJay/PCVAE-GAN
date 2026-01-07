import os
import torch
from omegaconf import OmegaConf

from train.st1_vae import train_stage1
from train.st2_latent_gan import train_stage2
from train.st3_wrap import train_stage3

def main(config_path="config/pcvae_gan.yaml", device="cuda"):
    # ---------------------------
    # 1. 加载配置
    # ---------------------------
    cfg = OmegaConf.load(config_path)
    print("Config loaded:")
    print(cfg)

    checkpoint_base = cfg['train']['checkpoint_dir']
    os.makedirs(checkpoint_base, exist_ok=True)

    # ---------------------------
    # 2. Stage1: VAE pretrain (simulation data)
    # ---------------------------
    print("========== Stage1: VAE Pretraining ==========")
    stage1_ckpt_dir = os.path.join(checkpoint_base, "stage1")
    os.makedirs(stage1_ckpt_dir, exist_ok=True)
    encoder_s1, decoder_s1 = train_stage1(cfg, checkpoint_dir=stage1_ckpt_dir, device=device)

    # ---------------------------
    # 3. Stage2: Latent GAN (latent alignment)
    # ---------------------------
    print("========== Stage2: Latent GAN Alignment ==========")
    stage2_ckpt_dir = os.path.join(checkpoint_base, "stage2")
    os.makedirs(stage2_ckpt_dir, exist_ok=True)
    encoder_s2, discriminator_s2 = train_stage2(cfg, stage1_ckpt_dir=stage1_ckpt_dir,
                                                checkpoint_dir=stage2_ckpt_dir, device=device)

    # ---------------------------
    # 4. Stage3: Physical consistency (real data)
    # ---------------------------
    print("========== Stage3: Real Data Wrap Consistency ==========")
    stage3_ckpt_dir = os.path.join(checkpoint_base, "stage3")
    os.makedirs(stage3_ckpt_dir, exist_ok=True)
    encoder_s3, decoder_s3 = train_stage3(cfg, stage2_ckpt_dir=stage2_ckpt_dir,
                                          checkpoint_dir=stage3_ckpt_dir, device=device)

    print("All stages finished. Final models saved in:")
    print(f"Stage1: {stage1_ckpt_dir}")
    print(f"Stage2: {stage2_ckpt_dir}")
    print(f"Stage3: {stage3_ckpt_dir}")

    return encoder_s3, decoder_s3, discriminator_s2

if __name__ == "__main__":
    main()
