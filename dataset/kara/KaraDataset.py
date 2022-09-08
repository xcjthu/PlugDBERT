import json
import os
from torch.utils.data import Dataset
import kara_storage

def make_kara_dataset(config, mode, encoding="utf8", *args, **params):
    storage = kara_storage.KaraStorage("file://%s" % config.get("data", "%s_data_path" % mode))

    dataset = storage.open_dataset(config.get("data", "kara_namespace"), config.get("data", "kara_dataset"), "r", version=config.get("data", "kara_version"))
    ret = kara_storage.make_torch_dataset(dataset, shuffle=True)
    ret.length = len(dataset)
    return ret

