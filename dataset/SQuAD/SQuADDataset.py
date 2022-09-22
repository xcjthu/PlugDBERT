import json
import os
from torch.utils.data import Dataset

class SQuADDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        data = json.load(open(config.get("data", "%s_data_path" % mode), "r", encoding="utf8"))
        self.qas = []
        self.context = []
        for doc in data["data"]:
            for para in doc["paragraphs"]:
                context = para["context"]
                self.context.append(context)
                qas = []
                for qa in para["qas"]:
                    qa.update({"context": len(self.context) - 1})
                    # if "is_impossible" in qa and qa["is_impossible"]:
                    #     qa["answers"] = [{"text": "no answer"}]
                    qas.append(qa)
                self.qas.extend(qas)

    def __getitem__(self, idx):
        qa = self.qas[idx]
        ret = qa.copy()
        ret["context"] = self.context[qa["context"]]
        return ret

    def __len__(self):
        # return 160
        return len(self.qas)