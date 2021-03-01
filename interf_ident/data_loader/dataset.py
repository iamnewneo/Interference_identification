import torch
import numpy as np
from scipy import signal
from torch.utils.data import Dataset


class InterfIdentDataset(Dataset):
    def __init__(self, X, y, preprocessing, transform):
        self.X = X
        self.y = y
        self.preprocessing = preprocessing
        self.transform = transform

    def get_cwt(self, X):
        CWT_WIDTH = 25
        # Root of Sum of Square of Each point 2D->1D conversion
        X = X ** 2
        X = X[:, 0] + X[:, 1]
        X = np.sqrt(X)
        widths = np.arange(1, CWT_WIDTH)
        cwt_signal = signal.cwt(X, signal.ricker, widths=widths)
        return cwt_signal

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.preprocessing is not None:
            if self.preprocessing == "cwt":
                X = self.get_cwt(X)
        if self.transform:
            X = self.transform(X)
        else:
            X = np.expand_dims(X, axis=0)
            X = torch.tensor(X, dtype=torch.float32)
        return {
            "X": X.type(torch.float32),
            "target": torch.tensor(y).type(torch.LongTensor),
        }
