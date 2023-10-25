import torch
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    """ Dataset class to work with scipy sparse matrix in PyTorch """
    def __init__(self, sparse):
        self.sparse = sparse
        
    def __len__(self):
        return self.sparse.shape[0]
        
    def __getitem__(self, idx):
        return torch.tensor(self.sparse[idx].todense())