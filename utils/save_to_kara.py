import json
import kara_storage
import os

out_path = "/data_new/private/xiaochaojun/DomainPlugin/data/pretrain/bio-kara"
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)


storage = kara_storage.KaraStorage("file://%s" % out_path)

dataset = storage.open_dataset("bio", "train", "w", version="lastest")
linenum = 0
for line in open("/data_new/private/xiaochaojun/DomainPlugin/train.txt", "r"):
    dataset.write(line.strip())
    linenum += 1
    if linenum % 100000 == 0:
        print(linenum)
dataset.close()
print("training num", linenum)

dataset = storage.open_dataset("bio", "valid", "w", version="lastest")
linenum = 0
for line in open("/data_new/private/xiaochaojun/DomainPlugin/valid.txt", "r"):
    dataset.write(line.strip())
    linenum += 1
    if linenum % 100000 == 0:
        print(linenum)
dataset.close()
print("valid num", linenum)

