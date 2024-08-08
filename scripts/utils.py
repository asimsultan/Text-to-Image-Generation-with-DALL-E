
import torch
from torch.utils.data import Dataset
import pandas as pd

class TextImageDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['text']
        self.targets = data['image']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(data):
    preprocessed_data = {
        "text": data['text'].tolist(),
        "image": data['image'].tolist()
    }
    return preprocessed_data
