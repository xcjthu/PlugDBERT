import json
import os
from torch.utils.data import Dataset
import kara_storage

def make_kara_dataset(config, mode, encoding="utf8", *args, **params):
    storage = kara_storage.KaraStorage("file://%s" % config.get("data", "%s_data_path" % mode))

    dataset = storage.open_dataset(
        config.get("data", "%s_kara_namespace" % mode),
        config.get("data", "%s_kara_dataset" % mode), "r",
        version=config.get("data", "%s_kara_version" % mode))
    ret = kara_storage.make_torch_dataset(dataset, shuffle=True)
    ret.length = len(dataset)
    return ret

