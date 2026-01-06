import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SimDataset(Dataset):
    """
    模拟数据 Dataset
    输入：wrap 相位 x_wrap
    输出：连续相位 phi_gt
    """
    def __init__(self, data_dir):
        self.files = os.listdir(data_dir)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(path)  # shape: (2,H,W) -> x_wrap, phi_gt
        x_wrap = torch.tensor(data[0], dtype=torch.float32).unsqueeze(0)
        phi_gt = torch.tensor(data[1], dtype=torch.float32).unsqueeze(0)
        return x_wrap, phi_gt
