from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k].tolist() for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]