import os
import torch
from torch.utils.data import Dataset
import numpy as np

class PhaseDataset(Dataset):
    """
    支持模拟数据 (x, phi) 和真实数据 (x, None)
    """
    def __init__(self, wrap_dir, phi_dir=None, mode='sim', transform=None):
        """
        wrap_dir: 缠绕相位路径
        phi_dir: 连续相位路径（仅模拟数据需要）
        mode: 'sim' 或 'real'
        transform: 可选 transform，例如归一化
        """
        self.wrap_dir = wrap_dir
        self.phi_dir = phi_dir
        self.mode = mode
        self.transform = transform

        self.wrap_files = sorted(os.listdir(wrap_dir))
        if mode == 'sim':
            assert phi_dir is not None, "模拟数据需要提供 phi_dir"
            self.phi_files = sorted(os.listdir(phi_dir))
            assert len(self.wrap_files) == len(self.phi_files), "模拟 x 与 phi 文件数不一致"

    def __len__(self):
        return len(self.wrap_files)

    def __getitem__(self, idx):
        # 加载 wrap 相位
        x = np.load(os.path.join(self.wrap_dir, self.wrap_files[idx])).astype(np.float32)
        x = torch.from_numpy(x)

        phi = None
        if self.mode == 'sim':
            phi = np.load(os.path.join(self.phi_dir, self.phi_files[idx])).astype(np.float32)
            phi = torch.from_numpy(phi)

        if self.transform:
            x = self.transform(x)
            if phi is not None:
                phi = self.transform(phi)

        return x, phi
