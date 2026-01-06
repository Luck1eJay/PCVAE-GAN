import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RealDataset(Dataset):
    """
    真实 wrap 数据 Dataset
    输入：wrap 相位 x_wrap
    无 phi_gt
    """
    def __init__(self, data_dir):
        self.files = os.listdir(data_dir)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        x_wrap = np.load(path)
        x_wrap = torch.tensor(x_wrap, dtype=torch.float32).unsqueeze(0)
        return x_wrap
