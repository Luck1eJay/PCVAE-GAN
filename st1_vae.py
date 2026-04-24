import os
from typing import Any, Mapping
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import PhaseDataset
from losses.vae_loss import vae_loss
from models.vae import NVAE


def _pick(cfg: Mapping[str, Any], *keys, default=None):
    for key in keys:
        if key in cfg and cfg[key] not in (None, ""):
            return cfg[key]
    return default


def _resolve_device(device):
    requested = device or 'cuda'
    requested_str = str(requested)
    if requested_str.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU instead of '{}'".format(requested_str))
        return torch.device('cpu')
    try:
        return torch.device(requested_str)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("Invalid device setting: {}".format(requested_str))


def _load_stage1_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return False
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        return False

    state_dict = None
    load_target = 'model'
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'encoder_state_dict' in ckpt or 'decoder_state_dict' in ckpt:
        load_target = 'encoder/decoder'
        if 'encoder_state_dict' in ckpt:
            enc_result = model.encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
            if getattr(enc_result, 'missing_keys', None) or getattr(enc_result, 'unexpected_keys', None):
                print(
                    "Stage1 encoder checkpoint compatibility: missing={}, unexpected={}".format(
                        len(enc_result.missing_keys), len(enc_result.unexpected_keys)
                    )
                )
        if 'decoder_state_dict' in ckpt:
            dec_result = model.decoder.load_state_dict(ckpt['decoder_state_dict'], strict=False)
            if getattr(dec_result, 'missing_keys', None) or getattr(dec_result, 'unexpected_keys', None):
                print(
                    "Stage1 decoder checkpoint compatibility: missing={}, unexpected={}".format(
                        len(dec_result.missing_keys), len(dec_result.unexpected_keys)
                    )
                )
        print("Loaded Stage1 encoder/decoder checkpoint from {}".format(checkpoint_path))
        return True

    if state_dict is None:
        return False

    load_result = model.load_state_dict(state_dict, strict=False)
    if getattr(load_result, 'missing_keys', None) or getattr(load_result, 'unexpected_keys', None):
        print(
            "Stage1 {} checkpoint compatibility: missing={}, unexpected={}".format(
                load_target, len(load_result.missing_keys), len(load_result.unexpected_keys)
            )
        )
    print("Loaded Stage1 model checkpoint from {}".format(checkpoint_path))
    return True


def train_stage1(cfg, checkpoint_dir, device='cuda', n_samples=5):
    """Stage1: compatibility VAE training with optional hierarchical + z5 VQ."""
    train_cfg = cfg['train']
    data_cfg = cfg['data']
    loss_cfg = cfg['loss']
    model_cfg = cfg.get('model', {})
    batch_size = int(train_cfg.get('batch_size', 1))
    lr_encoder = float(train_cfg.get('lr_encoder', train_cfg.get('lr', 1e-4)))
    lr_decoder = float(train_cfg.get('lr_decoder', train_cfg.get('lr', 1e-4)))
    num_epochs = int(train_cfg.get('num_epochs_stage1', 1))
    latent_channels = int(model_cfg.get('latent_channels', model_cfg.get('latent_dim', 64)))
    latent_levels = int(model_cfg.get('latent_levels', 5))
    base_channels = int(model_cfg.get('base_channels', 32))
    output_channels = int(model_cfg.get('output_channels', 1))
    strict_hierarchical = bool(model_cfg.get('strict_hierarchical', True))
    legacy_output = bool(model_cfg.get('legacy_output', True))
    use_z5_vq = bool(model_cfg.get('use_z5_vq', False))
    vq_num_embeddings = int(model_cfg.get('vq_num_embeddings', 256))
    vq_commitment_cost = float(model_cfg.get('vq_commitment_cost', 0.25))
    sim_wrap_dir = _pick(data_cfg, 'sim_wrap', 'sim_data')
    sim_true_dir = _pick(data_cfg, 'sim_true', 'sim_phi')
    if not sim_wrap_dir or not sim_true_dir:
        raise KeyError('Stage1 requires data.sim_wrap/sim_true (or legacy sim_data/sim_phi)')
    wrap_key = data_cfg.get('wrap_key', 'input')
    phi_key = data_cfg.get('phi_key', 'output')
    dataset = PhaseDataset(
        wrap_dir=sim_wrap_dir,
        phi_dir=sim_true_dir,
        mode='sim',
        transform=None,
        wrap_key=wrap_key,
        phi_key=phi_key,
    )
    if len(dataset) == 0:
        raise ValueError("Stage1 dataset is empty: wrap_dir={}, phi_dir={}".format(sim_wrap_dir, sim_true_dir))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = _resolve_device(device)
    model = NVAE(
        input_channels=1,
        base_channels=base_channels,
        latent_channels=latent_channels,
        latent_levels=latent_levels,
        output_channels=output_channels,
        strict_hierarchical=strict_hierarchical,
        legacy_output=legacy_output,
        use_z5_vq=use_z5_vq,
        vq_num_embeddings=vq_num_embeddings,
        vq_commitment_cost=vq_commitment_cost,
    ).to(device)
    if lr_decoder == lr_encoder:
        optimizer = optim.Adam(model.parameters(), lr=lr_encoder)
    else:
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': lr_encoder},
            {'params': model.decoder.parameters(), 'lr': lr_decoder},
        ])
    init_ckpt = _pick(train_cfg, 'stage1_init_ckpt', default='')
    _load_stage1_checkpoint(model, init_ckpt, device)
    beta = float(loss_cfg.get('beta', 1e-4))
    kl_weights = loss_cfg.get('kl_weights', None)
    free_bits = float(loss_cfg.get('free_bits', 0.0))
    lambda_vq = float(loss_cfg.get('lambda_vq', 1.0))
    save_interval = int(train_cfg.get('save_interval_stage1', 10))
    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_geo = 0.0
        total_kl = 0.0
        total_vq = 0.0
        total_vq_ppl = 0.0
        batch_count = 0
        for x_sim, phi_sim in loader:
            x_sim = x_sim.to(device)
            phi_sim = phi_sim.to(device)
            phi_hat, post_mus, post_logvars, prior_mus, prior_logvars = model(x_sim, return_stats=True)
            vq_loss = model.get_last_vq_loss(default_device=phi_hat.device, default_dtype=phi_hat.dtype)
            if vq_loss is None:
                vq_loss = torch.zeros((), device=phi_hat.device, dtype=phi_hat.dtype)
            loss, loss_geo, loss_kl, kl_levels = vae_loss(
                phi_hat,
                phi_sim,
                post_mus,
                post_logvars,
                lambda_geo=1.0,
                lambda_kl=beta,
                prior_mu=prior_mus,
                prior_logvar=prior_logvars,
                free_bits=free_bits,
                kl_weights=kl_weights,
                vq_loss=vq_loss,
                lambda_vq=lambda_vq,
                return_kl_levels=True,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_geo += float(loss_geo.item())
            total_kl += float(loss_kl.item())
            total_vq += float(vq_loss.item())
            vq_ppl = model.get_last_vq_perplexity()
            if vq_ppl is not None:
                total_vq_ppl += float(vq_ppl.item())
            batch_count += 1
        if batch_count == 0:
            print("[Stage1] Epoch {}/{}, no batches found.".format(epoch, num_epochs))
            continue
        avg_loss = total_loss / batch_count
        avg_geo = total_geo / batch_count
        avg_kl = total_kl / batch_count
        avg_vq = total_vq / batch_count
        avg_vq_ppl = total_vq_ppl / batch_count if total_vq_ppl > 0 else 0.0
        print(
            "[Stage1] Epoch {}/{}, Loss: {:.6f}, Geo: {:.6f}, KL: {:.6f}, VQ: {:.6f}, VQ_PPL: {:.3f}".format(
                epoch, num_epochs, avg_loss, avg_geo, avg_kl, avg_vq, avg_vq_ppl
            )
        )
        if epoch % save_interval == 0 or avg_loss < best_loss:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': min(best_loss, avg_loss),
                },
                checkpoint_path,
            )
            print("Checkpoint saved: {}".format(checkpoint_path))
            best_loss = min(best_loss, avg_loss)
    return model.encoder, model.decoder
if __name__ == '__main__':
    from omegaconf import OmegaConf
    config_path = 'config/pcvae_gan.yaml'
    cfg = OmegaConf.load(config_path)
    print('Config loaded:')
    print(cfg)
    checkpoint_base = cfg['train']['checkpoint_dir']
    os.makedirs(checkpoint_base, exist_ok=True)
    device = _resolve_device(cfg['train'].get('device', 'cuda:0'))
    print('Using device: {}'.format(device))
    stage1_ckpt_dir = os.path.join(checkpoint_base, 'stage1')
    os.makedirs(stage1_ckpt_dir, exist_ok=True)
    train_stage1(cfg, checkpoint_dir=stage1_ckpt_dir, device=str(device))
