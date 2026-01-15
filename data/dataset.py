from scipy.io import loadmat
import os
import torch
from torch.utils.data import Dataset


class PhaseDataset(Dataset):
    """
    基于 .mat 文件的相位展开数据集加载器。

    支持两种模式：
    1. 模拟数据（sim）：从 `wrapped` 和 `labels` 两个子目录加载缠绕相位与真实值。
    2. 无真值数据（real）：从 `wrapped` 子目录加载真实数据。
    """

    def __init__(self, base_dir, mode='sim', transform=None, key_wrap="wrapped", key_phi="labels"):
        """
        Args:
            base_dir (str): 主数据文件夹路径，例如 'dataset/train'。
            mode (str): 数据模式，'sim'（带真值）或 'real'（无真值）。
            transform (callable): 数据变换，用于标准化或归一化处理。
            key_wrap (str): 缠绕相位在 .mat 文件中的键名。
            key_phi (str): 连续相位在 .mat 文件中的键名（mode='sim'时需提供）。
        """
        self.wrap_dir = os.path.join(base_dir, "wrapped")
        self.phi_dir = os.path.join(base_dir, "labels") if mode == 'sim' else None
        self.mode = mode
        self.transform = transform
        self.key_wrap = key_wrap
        self.key_phi = key_phi

        # 载入文件名
        self.wrap_files = sorted(os.listdir(self.wrap_dir))
        if mode == 'sim':  # 模拟数据才需要labels文件夹
            assert self.phi_dir, "模拟数据设置必须提供连续相位路径."
            self.phi_files = sorted(os.listdir(self.phi_dir))
            assert len(self.wrap_files) == len(self.phi_files), "wrapped 与 labels 文件数不匹配！"

    def __len__(self):
        return len(self.wrap_files)

    def __getitem__(self, idx):
        # 读取缠绕相位
        wrap_file = os.path.join(self.wrap_dir, self.wrap_files[idx])
        wrap_mat = loadmat(wrap_file)
        x_wrap = torch.tensor(wrap_mat[self.key_wrap], dtype=torch.float32)  # 加载为 Tensor

        x_phi = None
        if self.mode == 'sim':
            label_file = os.path.join(self.phi_dir, self.phi_files[idx])
            phi_mat = loadmat(label_file)
            x_phi = torch.tensor(phi_mat[self.key_phi], dtype=torch.float32)  # 加载为 Tensor

        # 应用数据变换（如果提供）
        if self.transform:
            x_wrap = self.transform(x_wrap)
            if x_phi is not None:
                x_phi = self.transform(x_phi)

        return x_wrap, x_phi