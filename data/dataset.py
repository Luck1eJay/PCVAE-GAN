import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat

class PhaseDataset(Dataset):
    """
    支持模拟数据 (x, phi) 和真实数据 (x, None)
    """
    def __init__(self, wrap_dir, phi_dir=None, mode='sim', transform=None,
                 wrap_key='input', phi_key='output'):
        """
        wrap_dir: 缠绕相位路径（.mat）
        phi_dir: 连续相位路径（.mat，仅模拟数据需要）
        mode: 'sim' 或 'real'
        transform: 可选 transform，例如归一化
        wrap_key: wrap .mat 中数据字段名
        phi_key: phi .mat 中数据字段名
        """
        self.wrap_dir = wrap_dir
        self.phi_dir = phi_dir
        self.mode = mode
        self.transform = transform
        self.wrap_key = wrap_key
        self.phi_key = phi_key

        # 仅保留 .mat 文件，并按数值顺序排序
        self.wrap_files = self._sorted_mat_files(wrap_dir)

        if mode == 'sim':
            assert phi_dir is not None, "模拟数据需要提供 phi_dir"
            self.phi_files = self._sorted_mat_files(phi_dir)
            assert len(self.wrap_files) == len(self.phi_files), "模拟 x 与 phi 文件数不一致"

    def _sorted_mat_files(self, directory):
        files = [f for f in os.listdir(directory) if f.lower().endswith('.mat')]
        # 依据文件名中的数字排序，如 000001.mat, 000002.mat
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return files

    def __len__(self):
        return len(self.wrap_files)

    def _load_mat(self, file_path, key):
        mat = loadmat(file_path)
        if key not in mat:
            raise KeyError(f"在 {file_path} 中找不到字段 '{key}'")
        data = mat[key]
        # 确保 float32

        return torch.from_numpy(np.array(data, dtype=np.float32))

    def __getitem__(self, idx):
        # 加载 wrap 相位
        wrap_path = os.path.join(self.wrap_dir, self.wrap_files[idx])
        x = self._load_mat(wrap_path, self.wrap_key)

        phi = None
        if self.mode == 'sim':
            phi_path = os.path.join(self.phi_dir, self.phi_files[idx])
            phi = self._load_mat(phi_path, self.phi_key)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if phi is not None and phi.dim() == 2:
            phi = phi.unsqueeze(0)

        if self.transform:
            x = self.transform(x)
            if phi is not None:
                phi = self.transform(phi)

        return x, phi