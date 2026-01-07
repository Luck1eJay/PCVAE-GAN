import torch
import yaml

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
def wrap_phase(phi):
    """
    Wrap连续相位到 [-π, π]
    wrap(phi) = atan2(sin(phi), cos(phi))
    """
    return torch.atan2(torch.sin(phi), torch.cos(phi))

def unwarp_phase(phi_wrap, phi_prev=None):
    """
    简单一维解缠示例，可扩展为二维
    phi_unwrap[n] = phi_wrap[n] + 2π * round((phi_prev - phi_wrap[n]) / 2π)
    """
    if phi_prev is None:
        return phi_wrap
    k = torch.round((phi_prev - phi_wrap)/(2*torch.pi))
    return phi_wrap + 2*torch.pi*k
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cuda"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
def load_cfg(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)
def sample_multiple_solutions(vae, x_wrap, N=5):
    solutions = []
    mu, logvar = vae.encoder(x_wrap)
    for _ in range(N):
        z = reparameterize(mu, logvar)
        phi_hat = vae.decoder(x_wrap, z)
        solutions.append(phi_hat)
    return solutions
