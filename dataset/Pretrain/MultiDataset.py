import json
import os
from torch.utils.data import Dataset
import kara_storage
from torch.utils.data import DataLoader

class MultiDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        storage = kara_storage.KaraStorage("file:///data3/private/lirun/biomed/kara")
        self.dataset1 = storage.open_dataset("bio", mode, "r", version="latest")
        # self.dataset1 = kara_storage.make_torch_dataset(dataset, shuffle=True)
        # self.dataset1.length = len(dataset)
        self.epochnum1 = 0

        storage2 = kara_storage.KaraStorage("file:///data/disk1/private/xcj/DomainPlugin/data/pretrain/wiki-kara")
        self.dataset2 = storage2.open_dataset("wiki", mode, "r", version="latest")
        # self.dataset2 = kara_storage.make_torch_dataset(dataset2, shuffle=True)
        # self.dataset2.length = len(dataset2)
        self.epochnum2 = 0

        # self.generator1 = self.get_data_from1()
        # self.generator2 = self.get_data_from2()

    def get_data_from1(self):
        ret = self.dataset1.read()
        if ret is None:
            self.dataset1.seek(0, 0)
            ret = self.dataset1.read()
        return ret

    def get_data_from2(self):
        ret = self.dataset2.read()
        if ret is None:
            self.dataset2.seek(0, 0)
            ret = self.dataset2.read()
        return ret

    def __getitem__(self, idx):
        data1, data2 = self.get_data_from1(), self.get_data_from2()
        return data1, data2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

