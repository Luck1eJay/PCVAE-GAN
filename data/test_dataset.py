import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PhaseTestDataset(Dataset):
    """
    Test dataset for phase unwrapping inference

    - Simulation mode:
        input: wrapped phase
        target: true phase
    - Real mode:
        input: wrapped phase
        target: None
    """

    def __init__(self, wrap_dir, true_dir=None):
        self.wrap_dir = wrap_dir
        self.true_dir = true_dir

        self.wrap_files = sorted([
            f for f in os.listdir(wrap_dir) if f.endswith(".npy")
        ])

        if self.true_dir is not None:
            self.true_files = sorted([
                f for f in os.listdir(true_dir) if f.endswith(".npy")
            ])
            assert len(self.wrap_files) == len(self.true_files)
        else:
            self.true_files = None

    def __len__(self):
        return len(self.wrap_files)

    def __getitem__(self, idx):
        name = self.wrap_files[idx]

        wrap = np.load(os.path.join(self.wrap_dir, name)).astype(np.float32)
        wrap = torch.from_numpy(wrap).unsqueeze(0)  # [1, H, W]

        if self.true_dir is not None:
            true = np.load(os.path.join(self.true_dir, name)).astype(np.float32)
            true = torch.from_numpy(true).unsqueeze(0)
            return wrap, true, name
        else:
            return wrap, name
