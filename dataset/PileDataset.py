import json
import os
from torch.utils.data import Dataset
import datasets

class PileDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        # self.data = datasets.load_from_disk(config.get("data", "data_path"))[mode]

    def __getitem__(self, idx):
        # return self.data[idx]
        return {"text": "test " * 200}

    def __len__(self):
        # return len(self.data)
        return 10000

