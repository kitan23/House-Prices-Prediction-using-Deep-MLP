import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HousePriceDataset(Dataset): 
    def __init__(self, features, labels): 
      self.features = torch.tensor(features, dtype=torch.float32)
      self.labels = torch.tensor(labels.values, dtype=torch.float32) 
    def __len__(self):
      return len(self.labels)
    def __getitem__(self, idx):
      return self.features[idx], self.labels[idx]
