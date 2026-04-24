import glob
import math
import os

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import PhaseDataset
from data.vaedata import UnwrappedPhaseDataset
from models.vae import NVAE
from utils.metrics import psnr, rmse, ssim


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_weight_path(cfg):
    test_cfg = cfg.get('test', {})
    for key in ('model_path', 'checkpoint'):
        if test_cfg.get(key):
            return test_cfg[key]

    checkpoint_dir = cfg.get('train', {}).get('checkpoint_dir', 'checkpointsVAE2_vq')
    stage1_dir = os.path.join(checkpoint_dir, 'stage1')
    candidates = sorted(glob.glob(os.path.join(stage1_dir, 'checkpoint_epoch*.pth')))
    if candidates:
        return candidates[-1]
    return os.path.join(stage1_dir, 'checkpoint_epoch10.pth')


def _load_model_weights(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            return
        if 'encoder_state_dict' in ckpt and 'decoder_state_dict' in ckpt:
            model.encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
            model.decoder.load_state_dict(ckpt['decoder_state_dict'], strict=False)
            return
    model.load_state_dict(ckpt, strict=False)


def _wrap_to_pi(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


def save_compare_plot(x_wrap, phi_gt, phi_hat, idx, save_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping comparison plots.')
        return

    os.makedirs(save_dir, exist_ok=True)
    gt_vmin = min(phi_gt.min().item(), phi_hat.min().item())
    gt_vmax = max(phi_gt.max().item(), phi_hat.max().item())
    err = (phi_hat - phi_gt).abs()
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(x_wrap, cmap='rainbow', vmin=-math.pi, vmax=math.pi)
    plt.title('Input wrapped')
    plt.colorbar()
    plt.subplot(1, 4, 2)
    plt.imshow(phi_gt, cmap='rainbow', vmin=gt_vmin, vmax=gt_vmax)
    plt.title('GT unwrapped')
    plt.colorbar()
    plt.subplot(1, 4, 3)
    plt.imshow(phi_hat, cmap='rainbow', vmin=gt_vmin, vmax=gt_vmax)
    plt.title('Pred unwrapped')
    plt.colorbar()
    plt.subplot(1, 4, 4)
    plt.imshow(err, cmap='magma')
    plt.title('|Error|')
    plt.colorbar()
    plt.suptitle(f'Test Sample {idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'compare_test_{idx}.png'))
    plt.close()


@torch.no_grad()
def test():
    cfg = load_config('config/pcvae_gan.yaml')
    test_cfg = cfg.get('test', {})
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})

    device = torch.device(test_cfg.get('device', train_cfg.get('device', 'cuda:0')))
    model = NVAE(
        input_channels=1,
        base_channels=model_cfg.get('base_channels', 32),
        latent_channels=model_cfg.get('latent_channels', model_cfg.get('latent_dim', 64)),
        latent_levels=model_cfg.get('latent_levels', 5),
        output_channels=model_cfg.get('output_channels', 1),
        strict_hierarchical=bool(model_cfg.get('strict_hierarchical', True)),
        legacy_output=True,
        use_z5_vq=bool(model_cfg.get('use_z5_vq', False)),
        vq_num_embeddings=model_cfg.get('vq_num_embeddings', 256),
        vq_commitment_cost=model_cfg.get('vq_commitment_cost', 0.25),
    ).to(device)

    weight_path = _resolve_weight_path(cfg)
    _load_model_weights(model, weight_path, device)
    model.eval()

    wrap_dir = (
        data_cfg.get('test_wrap')
        or data_cfg.get('test_wrapped')
        or data_cfg.get('test_wrap_dir')
        or data_cfg.get('sim_wrap')
    )
    phi_dir = (
        data_cfg.get('test_true')
        or data_cfg.get('test_absolute')
        or data_cfg.get('test_phi')
        or data_cfg.get('sim_true')
    )
    if wrap_dir and phi_dir and os.path.isdir(wrap_dir) and os.path.isdir(phi_dir):
        dataset = PhaseDataset(
            wrap_dir=wrap_dir,
            phi_dir=phi_dir,
            mode='sim',
            transform=None,
            wrap_key=data_cfg.get('wrap_key', 'input'),
            phi_key=data_cfg.get('phi_key', 'output'),
        )
        paired_mode = True
    else:
        if not phi_dir:
            raise KeyError(
                'config must provide paired test data: data.test_wrap/test_true, '
                'data.test_wrap/test_absolute, or at least data.test_phi for fallback'
            )
        dataset = UnwrappedPhaseDataset(
            phi_dir=phi_dir,
            phi_key=data_cfg.get('phi_key', 'output'),
            transform=None,
        )
        paired_mode = False
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    save_dir = os.path.join(test_cfg.get('out_dir', 'test_out'))
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 20)
    print('Test mode: {}'.format('paired wrapped -> unwrapped' if paired_mode else 'unwrapped fallback'))
    print('Checkpoint: {}'.format(weight_path))
    print('Output dir: {}'.format(save_dir))
    print('=' * 20)

    total_mae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_wrap_mae = 0.0
    sample_num = 0
    summary_path = os.path.join(save_dir, 'metrics.txt')
    summary_lines = []

    for idx, batch in enumerate(dataloader):
        if paired_mode:
            x_wrap, phi_gt = batch
            x_wrap = x_wrap.to(device)
            phi_gt = phi_gt.to(device)
        else:
            phi_gt = batch.to(device)
            x_wrap = phi_gt

        outputs = model(x_wrap)
        if isinstance(outputs, tuple):
            phi_hat = outputs[0]
        else:
            phi_hat = outputs

        mae_val = torch.mean(torch.abs(phi_hat - phi_gt)).item()
        rmse_val = float(rmse(phi_hat, phi_gt).item())
        dyn_range = float((phi_gt.max() - phi_gt.min()).item())
        psnr_val = float(psnr(phi_hat, phi_gt, max_val=dyn_range if dyn_range > 0 else 1.0).item())
        ssim_val = float(ssim(phi_hat, phi_gt).item())
        wrap_mae_val = float(torch.mean(torch.abs(_wrap_to_pi(phi_hat) - x_wrap)).item()) if paired_mode else None

        pred = phi_hat[0, 0].detach().cpu().numpy()
        gt = phi_gt[0, 0].detach().cpu().numpy()
        x_vis = x_wrap[0, 0].detach().cpu().numpy()
        save_compare_plot(x_vis, gt, pred, idx, save_dir)

        total_mae += mae_val
        total_rmse += rmse_val
        total_psnr += psnr_val
        total_ssim += ssim_val
        if wrap_mae_val is not None:
            total_wrap_mae += wrap_mae_val
        sample_num += 1
        if idx < 5:
            if wrap_mae_val is None:
                print(
                    f'Sample {idx}: MAE = {mae_val:.6f}, RMSE = {rmse_val:.6f}, '
                    f'PSNR = {psnr_val:.3f}, SSIM = {ssim_val:.4f}, Wrap-MAE = N/A'
                )
            else:
                print(
                    f'Sample {idx}: MAE = {mae_val:.6f}, RMSE = {rmse_val:.6f}, '
                    f'PSNR = {psnr_val:.3f}, SSIM = {ssim_val:.4f}, Wrap-MAE = {wrap_mae_val:.6f}'
                )

        if wrap_mae_val is None:
            summary_lines.append(
                f'Sample {idx}: MAE={mae_val:.6f}, RMSE={rmse_val:.6f}, '
                f'PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}, Wrap-MAE=N/A\n'
            )
        else:
            summary_lines.append(
                f'Sample {idx}: MAE={mae_val:.6f}, RMSE={rmse_val:.6f}, '
                f'PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}, Wrap-MAE={wrap_mae_val:.6f}\n'
            )

    print('=' * 20)
    print(f'Mean MAE (all test): {total_mae / sample_num:.6f}')
    print(f'Mean RMSE (all test): {total_rmse / sample_num:.6f}')
    print(f'Mean PSNR (all test): {total_psnr / sample_num:.6f}')
    print(f'Mean SSIM (all test): {total_ssim / sample_num:.6f}')
    if paired_mode:
        print(f'Mean Wrap-MAE (all test): {total_wrap_mae / sample_num:.6f}')
    else:
        print('Mean Wrap-MAE (all test): N/A')
    print(f'重建结果图像保存在: {save_dir}/')
    print('=' * 20)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Mean MAE: {total_mae / sample_num:.6f}\n')
        f.write(f'Mean RMSE: {total_rmse / sample_num:.6f}\n')
        f.write(f'Mean PSNR: {total_psnr / sample_num:.6f}\n')
        f.write(f'Mean SSIM: {total_ssim / sample_num:.6f}\n')
        if paired_mode:
            f.write(f'Mean Wrap-MAE: {total_wrap_mae / sample_num:.6f}\n')
        else:
            f.write('Mean Wrap-MAE: N/A\n')
        f.writelines(summary_lines)


if __name__ == '__main__':
    test()
