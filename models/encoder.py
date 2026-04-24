import torch
import torch.nn as nn
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
class NVAEEncoder(nn.Module):
    """Five-level encoder that produces z1~z5 posterior parameters."""
    def __init__(self, input_channels=1, base_channels=32, latent_channels=64, latent_levels=5, *args, **kwargs):
        super(NVAEEncoder, self).__init__()
        if latent_levels != 5:
            raise ValueError('This compatibility implementation expects latent_levels=5')
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.latent_levels = int(latent_levels)
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        channel_multipliers = [1, 2, 4, 8, 8]
        feature_channels = [self.base_channels * m for m in channel_multipliers]
        self.feature_channels = feature_channels
        down_blocks = []
        in_ch = self.input_channels
        for out_ch in feature_channels:
            down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    _group_norm(out_ch),
                    nn.ReLU(inplace=True),
                    ResBlock(out_ch),
                )
            )
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mu_logvar_heads = nn.ModuleList([
            nn.Conv2d(ch, self.latent_channels * 2, 1) for ch in feature_channels
        ])
    def forward(self, x):
        features = []
        h = x
        for block in self.down_blocks:
            h = block(h)
            features.append(h)
        mus, logvars = [], []
        for feat, head in zip(features, self.mu_logvar_heads):
            mu_logvar = head(feat)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars
    def sample(self, mus, logvars, deterministic=False):
        if len(mus) != len(logvars):
            raise ValueError('mus and logvars must have the same length')
        zs = []
        for mu, logvar in zip(mus, logvars):
            if deterministic:
                zs.append(mu)
            else:
                std = torch.exp(0.5 * logvar)
                zs.append(mu + torch.randn_like(std) * std)
        return zs
# Legacy compatibility alias
Encoder = NVAEEncoder
