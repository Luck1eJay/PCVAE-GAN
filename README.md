PCVAE-GAN (Physics-Constrained VAE + Latent GAN) is a deep learning framework for 2D phase unwrapping. The method integrates supervised learning on simulated data, latent space alignment with a GAN, and physical consistency constraints to enable transfer from simulated to real data. The model outputs continuous phase maps, supports multi-solution sampling, and provides uncertainty estimation, all while preserving physical plausibility.

Key features:

Stage 1 (VAE): Learns phase unwrapping patterns on simulated data (supervised)

Stage 2 (Latent GAN): Aligns latent space distributions of simulated and real data

Stage 3 (Physical Consistency): Enforces wrap and gradient constraints on real data

Multi-solution sampling and uncertainty estimation

Optional self-supervised fine-tuning for new scenarios

Applications: optical imaging, SAR imaging, interferometry, and other tasks requiring 2D phase unwrapping.
