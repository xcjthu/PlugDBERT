import json
import os
from torch.utils.data import Dataset
from ffrecord import FileReader
import pickle

class FFRecordDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.reader = FileReader(config.get("data", "%s_data_path" % mode))
        # self.data = datasets.load_from_disk(config.get("data", "data_path"))[mode]

    def __getitem__(self, idx):
        # return self.data[idx]
        return {"text": pickle.loads(self.reader.read([idx])[0])}

    def __len__(self):
        # return len(self.data)
        # return 10000
        return self.reader.n

