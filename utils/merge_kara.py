import json
import kara_storage
import os
from tqdm import tqdm
from random import random

out_path = "/data/disk1/private/xcj/DomainPlugin/data/pretrain/wiki-bio-kara"
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)


for mode in ["train", "valid"]:
    storage = kara_storage.KaraStorage("file:///data3/private/lirun/biomed/kara")
    dataset = storage.open_dataset("bio", mode, "r", version="latest")
    storage2 = kara_storage.KaraStorage("file:///data/disk1/private/xcj/DomainPlugin/data/pretrain/wiki-kara")
    dataset2 = storage2.open_dataset("wiki", mode, "r", version="latest")

    storage_merge = kara_storage.KaraStorage("file://%s" % out_path)
    merge_data = storage_merge.open_dataset("wiki-bio", mode, "w", version="lastest")

    print(mode, "bio", len(dataset), "wiki", len(dataset2))
    for i in tqdm(range(max(len(dataset), len(dataset2)))):
        data0, data1 = dataset.read(), dataset2.read()
        if data0 is None:
            dataset.seek(0, 0)
            data0 = dataset.read()
        if data1 is None:
            dataset2.seek(0, 0)
            data1 = dataset2.read()
        merge_data.write([data0, data1])
    merge_data.close()
