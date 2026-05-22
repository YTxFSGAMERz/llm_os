import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Encode the entire text at once
        # For huge datasets, we would memmap. For tinyshakespeare, it fits in RAM.
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
    def __len__(self):
        # We need seq_len + 1 tokens for x and y
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(file_path, tokenizer, seq_len, batch_size, split="train", train_ratio=0.9):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    n = len(text)
    train_data = text[:int(n * train_ratio)]
    val_data = text[int(n * train_ratio):]

    data = train_data if split == "train" else val_data
    dataset = TextDataset(data, tokenizer, seq_len)
    
    # Shuffle only training data
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    return dataloader
