import os
from omegaconf import OmegaConf

from st1_vae import train_stage1
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

    finetune_cfg = cfg.get('finetune', {})
    use_finetune = bool(finetune_cfg.get('stage1_ckpt_path') or
                        finetune_cfg.get('stage2_ckpt_path') or
                        finetune_cfg.get('stage3_ckpt_path'))

    # ---------------------------
    # 2. Stage1: VAE pretrain (simulation data)
    # ---------------------------
    stage1_ckpt_dir = os.path.join(checkpoint_base, "stage1")
    os.makedirs(stage1_ckpt_dir, exist_ok=True)
    if use_finetune and finetune_cfg.get('stage1_ckpt_path'):
        print(f"========== Stage1: Fine-tuning VAE from {finetune_cfg['stage1_ckpt_path']} ==========")
        cfg['train']['lr'] = finetune_cfg.get('lr_encoder_finetune', cfg['train']['lr'])
        encoder_s1, decoder_s1 = train_stage1(cfg, checkpoint_dir=stage1_ckpt_dir, device=device)
    else:
        print("========== Stage1: VAE Pretraining ==========")
        encoder_s1, decoder_s1 = train_stage1(cfg, checkpoint_dir=stage1_ckpt_dir, device=device)

    # ---------------------------
    # 3. Stage2: Latent GAN (latent alignment)
    # ---------------------------
    stage2_ckpt_dir = os.path.join(checkpoint_base, "stage2")
    os.makedirs(stage2_ckpt_dir, exist_ok=True)
    if use_finetune and finetune_cfg.get('stage2_ckpt_path'):
        print(f"========== Stage2: Fine-tuning Latent GAN from {finetune_cfg['stage2_ckpt_path']} ==========")
        cfg['train']['lr_encoder'] = finetune_cfg.get('lr_encoder_finetune', cfg['train']['lr_encoder'])
        cfg['train']['lr_discriminator'] = finetune_cfg.get('lr_discriminator_finetune', cfg['train']['lr_discriminator'])
        encoder_s2, discriminator_s2 = train_stage2(cfg, stage1_ckpt_dir=stage1_ckpt_dir,
                                                    checkpoint_dir=stage2_ckpt_dir, device=device)
    else:
        print("========== Stage2: Latent GAN Alignment ==========")
        encoder_s2, discriminator_s2 = train_stage2(cfg, stage1_ckpt_dir=stage1_ckpt_dir,
                                                    checkpoint_dir=stage2_ckpt_dir, device=device)

    # ---------------------------
    # 4. Stage3: Physical consistency (real data)
    # ---------------------------
    stage3_ckpt_dir = os.path.join(checkpoint_base, "stage3")
    os.makedirs(stage3_ckpt_dir, exist_ok=True)
    if use_finetune and finetune_cfg.get('stage3_ckpt_path'):
        print(f"========== Stage3: Fine-tuning Physical Consistency from {finetune_cfg['stage3_ckpt_path']} ==========")
        cfg['train']['lr_encoder'] = finetune_cfg.get('lr_encoder_finetune', cfg['train']['lr_encoder'])
        cfg['train']['lr_decoder'] = finetune_cfg.get('lr_decoder_finetune', cfg['train']['lr_decoder'])
        if finetune_cfg.get('freeze_decoder', True):
            print("Decoder will be frozen during fine-tuning.")
        encoder_s3, decoder_s3 = train_stage3(cfg, stage2_ckpt_dir=stage2_ckpt_dir,
                                              checkpoint_dir=stage3_ckpt_dir, device=device)
    else:
        print("========== Stage3: Real Data Wrap Consistency ==========")
        encoder_s3, decoder_s3 = train_stage3(cfg, stage2_ckpt_dir=stage2_ckpt_dir,
                                              checkpoint_dir=stage3_ckpt_dir, device=device)

    print("All stages finished. Final models saved in:")
    print(f"Stage1: {stage1_ckpt_dir}")
    print(f"Stage2: {stage2_ckpt_dir}")
    print(f"Stage3: {stage3_ckpt_dir}")

    return encoder_s3, decoder_s3, discriminator_s2

if __name__ == "__main__":
    main()

