from torch.utils.data import DataLoader
from .dataset import PhaseDataset

def make_dataloader(wrap_dir, phi_dir=None, mode='sim', batch_size=16, shuffle=True, num_workers=4, transform=None):
    dataset = PhaseDataset(wrap_dir, phi_dir, mode=mode, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
