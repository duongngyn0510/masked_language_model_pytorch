from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np


class MyDataset(Dataset):
    def __init__(self, src_dir, tokenizer):
        super().__init__()
        with open(src_dir, 'r') as f:
            self.src = f.readlines()
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx].strip()