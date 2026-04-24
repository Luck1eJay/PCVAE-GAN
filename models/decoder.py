import torch
import torch.nn as nn
from models.vq import VectorQuantizer
def _group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            _group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            _group_norm(channels),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.block(x))
class NVAEDecoder(nn.Module):
    """Five-level hierarchical decoder with optional VQ on z5."""
    def __init__(
        self,
        latent_channels,
        base_channels,
        output_channels,
        latent_levels=5,
        use_z5_vq=False,
        vq_num_embeddings=256,
        vq_commitment_cost=0.25,
    ):
        super(NVAEDecoder, self).__init__()
        if latent_levels != 5:
            raise ValueError('This compatibility implementation expects latent_levels=5')
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.output_channels = int(output_channels)
        self.latent_levels = int(latent_levels)
        self.use_z5_vq = bool(use_z5_vq)
        self.vq_num_embeddings = int(vq_num_embeddings)
        self.vq_commitment_cost = float(vq_commitment_cost)
        # Channel ladder for 8 -> 16 -> 32 -> 64 -> 128 -> 256.
        self.feature_channels = [self.base_channels, self.base_channels * 2, self.base_channels * 4, self.base_channels * 8, self.base_channels * 8]
        self.top_channels = self.feature_channels[-1]
        self.top_proj = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.top_channels, 3, padding=1),
            _group_norm(self.top_channels),
            nn.ReLU(inplace=True),
            ResBlock(self.top_channels),
        )
        up_blocks = []
        fuse_blocks = []
        prior_heads = []
        for level in range(self.latent_levels - 1, 0, -1):
            in_ch = self.feature_channels[level]
            out_ch = self.feature_channels[level - 1]
            up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    _group_norm(out_ch),
                    nn.ReLU(inplace=True),
                    ResBlock(out_ch),
                )
            )
            fuse_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch + self.latent_channels, out_ch, 3, padding=1),
                    _group_norm(out_ch),
                    nn.ReLU(inplace=True),
                    ResBlock(out_ch),
                )
            )
            prior_heads.append(nn.Conv2d(out_ch, self.latent_channels * 2, 1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.fuse_blocks = nn.ModuleList(fuse_blocks)
        self.prior_heads = nn.ModuleList(prior_heads)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels[0], self.feature_channels[0], 4, stride=2, padding=1),
            _group_norm(self.feature_channels[0]),
            nn.ReLU(inplace=True),
            ResBlock(self.feature_channels[0]),
        )
        self.final_out = nn.Conv2d(self.feature_channels[0], self.output_channels, 3, padding=1)
        self.vq = VectorQuantizer(
            num_embeddings=self.vq_num_embeddings,
            embedding_dim=self.latent_channels,
            commitment_cost=self.vq_commitment_cost,
        ) if self.use_z5_vq else None
        self.last_vq_loss = None
        self.last_vq_perplexity = None
    def _maybe_quantize_top(self, z5):
        self.last_vq_loss = None
        self.last_vq_perplexity = None
        if self.vq is None:
            return z5
        quantized, vq_loss, stats = self.vq(z5)
        self.last_vq_loss = vq_loss
        self.last_vq_perplexity = stats.get('perplexity')
        return quantized
    def _decode_from_latents(self, zs):
        if not isinstance(zs, (list, tuple)):
            raise TypeError('zs must be a list/tuple of latent tensors')
        if len(zs) != self.latent_levels:
            raise ValueError('expected %d latent tensors, got %d' % (self.latent_levels, len(zs)))
        z5 = self._maybe_quantize_top(zs[-1])
        state = self.top_proj(z5)
        # Deep-to-shallow path: z5 -> z4 -> z3 -> z2 -> z1.
        for stage_idx, (up_block, fuse_block, prior_head) in enumerate(zip(self.up_blocks, self.fuse_blocks, self.prior_heads)):
            state = up_block(state)
            _ = prior_head(state)  # keeps a top-down prior hook for hierarchical NVAE
            level_idx = self.latent_levels - 2 - stage_idx
            state = fuse_block(torch.cat([state, zs[level_idx]], dim=1))
        state = self.final_up(state)
        return self.final_out(state)
    def forward(self, zs):
        return self._decode_from_latents(zs)
    def hierarchical_decode(self, post_mus, post_logvars, deterministic=False):
        if len(post_mus) != self.latent_levels or len(post_logvars) != self.latent_levels:
            raise ValueError('posterior lists must match latent_levels=5')
        zs = []
        for mu, logvar in zip(post_mus, post_logvars):
            if deterministic:
                zs.append(mu)
            else:
                std = torch.exp(0.5 * logvar)
                zs.append(mu + torch.randn_like(std) * std)
        out = self._decode_from_latents(zs)
        prior_mus = [None] * self.latent_levels
        prior_logvars = [None] * self.latent_levels
        prior_mus[-1] = torch.zeros_like(post_mus[-1])
        prior_logvars[-1] = torch.zeros_like(post_logvars[-1])
        state = self.top_proj(self._maybe_quantize_top(zs[-1]))
        for stage_idx, (up_block, prior_head) in enumerate(zip(self.up_blocks, self.prior_heads)):
            state = up_block(state)
            prior_mu, prior_logvar = torch.chunk(prior_head(state), 2, dim=1)
            level_idx = self.latent_levels - 2 - stage_idx
            prior_mus[level_idx] = prior_mu
            prior_logvars[level_idx] = prior_logvar
            state = self.fuse_blocks[stage_idx](torch.cat([state, zs[level_idx]], dim=1))
        return out, list(post_mus), list(post_logvars), prior_mus, prior_logvars
# Legacy compatibility alias
Decoder = NVAEDecoder
