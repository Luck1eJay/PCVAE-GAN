import torch
import torch.nn as nn


def r3gan_loss(D, z_real, z_sim):
    """
    Relativistic latent GAN loss
    这里判别器 D 只作用于 latent space (z)

    参数：
    - D: latent discriminator，输入 z 返回 logits
    - z_real: Encoder 对真实数据输出的 latent
    - z_sim: Encoder 对模拟数据输出的 latent

    返回:
    - loss_D: 判别器损失
    - loss_G: 生成器(Encoder)对抗损失
    """
    # 判别器输出
    D_real = D(z_sim)  # 模拟 latent 当作“真实”
    D_fake = D(z_real)  # 真实 latent 当作“伪”

    # Relativistic loss
    # 判别器
    loss_D = -torch.mean(torch.log(torch.sigmoid(D_real - D_fake.detach())))
    # 生成器 (Encoder)
    loss_G = -torch.mean(torch.log(torch.sigmoid(D_fake - D_real.detach())))

    return loss_D, loss_G


def gradient_penalty(D, z_real, z_sim, device='cuda'):
    """
    R1-style gradient penalty，用于 latent GAN 的稳定训练
    """
    alpha = torch.rand(z_real.size(0), 1).to(device)
    alpha = alpha.expand_as(z_real)

    interpolated = alpha * z_real + (1 - alpha) * z_sim
    interpolated.requires_grad_(True)

    D_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=D_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(D_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
