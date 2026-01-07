import torch
import torch.nn.functional as F

def diversity_loss(phi_samples):
    """
    Encourage multiple outputs per input to be different
    phi_samples: [batch, n_samples, H, W]
    """
    n_samples = phi_samples.size(1)
    if n_samples < 2:
        return torch.tensor(0.0, device=phi_samples.device)
    loss = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            loss += F.l1_loss(phi_samples[:, i], phi_samples[:, j])
    loss = -2 * loss / (n_samples * (n_samples - 1))  # normalize and negative for maximization
    return loss
