import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input = self.X[idx,:]
        target = self.y[idx]
        return (input, target)
