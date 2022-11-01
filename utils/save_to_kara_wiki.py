import json
import kara_storage
import os
from tqdm import tqdm
from random import random

out_path = "/data/disk1/private/xcj/DomainPlugin/data/pretrain/wiki-kara"
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)


storage = kara_storage.KaraStorage("file://%s" % out_path)

dataset = storage.open_dataset("wiki", "train", "w", version="lastest")
validdataset = storage.open_dataset("wiki", "valid", "w", version="lastest")

trainnum, validnum = 0, 0

for i in range(163):
    fin = open("/data/disk1/private/xcj/PLM_corpus/wiki/passages/%s.jsonl" % i, "r")
    for line in tqdm(fin.readlines()):
        data = json.loads(line)
        if not len(data["text"].split()) > 50:
            continue
        if random() < 0.002:
            validdataset.write(data)
            validnum += 1
        else:
            dataset.write(data)
            trainnum += 1
dataset.close()
validdataset.close()

print("train num", trainnum, "valid num", validnum )

