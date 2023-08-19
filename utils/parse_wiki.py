import json
import os
from tqdm import tqdm
import kara_storage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', '-out', default="data/c4-kara", required=True)
args = parser.parse_args()


out_path = args.out_path
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)

storage = kara_storage.KaraStorage("file://%s" % out_path)
dataset = storage.open("wiki", "train", "w", version="1st")

path = "/data/xiaochaojun/zh_gov_data/wiki/"
for folder in ["20230197", "20230198"]:
    for fname in tqdm(os.listdir(os.path.join(path, folder))):
        fin = open(os.path.join(path, folder, fname), "r")
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            if len(line) < 100:
                continue
            dataset.write(line)
dataset.close()

