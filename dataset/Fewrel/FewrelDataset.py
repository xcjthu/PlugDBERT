import enum
import json
import os
import random
from torch.utils.data import Dataset

class FewrelDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data = json.load(open(config.get("data", "%s_data_path" % mode), "r"))
        print("the number of data in %s: %s" % (mode, len(self.data)))
        self.rels = list(self.data.keys())
        self.nways = config.getint("train", "nways")
        self.nshot = config.getint("train", "nshot")
        self.nquery = config.getint("train", "nquery")

        
        sample_num = 5000 # if mode != "train" else 10000
        # if mode != "train":
        if True:
            random.seed(2333)
            self.sample_instance = []
            for i in range(sample_num):
                keys = random.sample(self.rels, self.nways)
                candidates = [random.sample(self.data[key], self.nshot + self.nquery) for key in keys]
                queries = [cand[:self.nquery] for cand in candidates]
                support = [cand[self.nquery:] for cand in candidates]
                for l, q in enumerate(queries):
                    instance = []
                    instance.extend(q)
                    instance.extend(support[l])
                    for i in range(len(support)):
                        if i == l:
                            continue
                        instance.extend(support[i])
                    self.sample_instance.append(instance)#{"instance": instance, "label": l})

                # qandc = random.sample(self.data[keys[0]], self.nshot + self.nquery)
                # negative = []
                # for r in keys[1:]:
                #     negative.extend(random.sample(self.data[r], self.nshot))
                # self.test_data.append(qandc + negative)

    def __getitem__(self, idx):
        # if self.mode == "train":
        #     keys = random.sample(self.rels, self.nways)
        #     # return [random.sample(self.data[key], self.nshot + self.nquery) for key in keys]
        #     query = random.sample(self.data[keys[0]], self.nshot + self.nquery)
        #     negative = []
        #     for r in keys[1:]:
        #         negative.extend(random.sample(self.data[r], self.nshot))
        #     return query + negative
        # else:
        return self.sample_instance[idx]
        
    def __len__(self):
        # if self.mode == "train":
        #     return len(self.data) * 1000
        # else:
        return len(self.sample_instance)