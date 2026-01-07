import torch

def r3gan_loss(D, z_real, z_sim):
    """
    Relativistic latent GAN loss
    D: latent discriminator
    z_real: Encoder 对真实数据 latent
    z_sim: Encoder 对模拟数据 latent
    """
    D_real = D(z_sim)    # 模拟 latent 当作“真实”
    D_fake = D(z_real)   # 真实 latent 当作“伪”

    # 判别器 loss
    loss_D = -torch.mean(torch.log(torch.sigmoid(D_real - D_fake.detach())))
    # Encoder 对抗 loss
    loss_G = -torch.mean(torch.log(torch.sigmoid(D_fake - D_real.detach())))
    return loss_D, loss_G

