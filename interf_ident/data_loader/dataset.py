import torch
from torch.utils.data import Dataset


class InterfIdentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return {
            "X": torch.tensor(X, dtype=torch.float32),
            "target": torch.tensor(y).type(torch.LongTensor),
        }
