import json
import torch
import os
import numpy as np

import random
from transformers import AutoTokenizer, T5Config
from nltk.tokenize import sent_tokenize

class FewrelFormatter:
    def __init__(self, config, mode, *args, **params):
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.plm = config.get("model", "pretrained_model")

        self.tokenizer = AutoTokenizer.from_pretrained(self.plm)
        self.nquery = config.getint("train", "nquery")

    def addmarker(self, instance):
        for pos in instance["h"][2]:
            instance["tokens"][pos[0]] = "*" + instance["tokens"][pos[0]]
            instance["tokens"][pos[-1]] = instance["tokens"][pos[-1]] + "*"
        for pos in instance["t"][2]:
            instance["tokens"][pos[0]] = "^" + instance["tokens"][pos[0]]
            instance["tokens"][pos[-1]] = instance["tokens"][pos[-1]] + "^"
        return self.tokenizer.encode(" ".join(instance["tokens"]))

    def padding(self, tokens):
        mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
        if len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        return tokens[:self.max_len], mask[:self.max_len]

    def process(self, data):
        inpids, mask = [], []
        # tokenized = []
        # for samples in data:
        #     tokenized.append([self.addmarker(ins) for ins in samples])
        # queries = [samples[:self.nquery] for samples in tokenized]
        # supporting = [samples[self.nquery:] for samples in tokenized]
        # for query in queries:
        #     for support in supporting:

        for samples in data:
            tokens = [self.addmarker(ins) for ins in samples]
            queries = tokens[:self.nquery]
            for q in queries:
                inps = [q + tokens[i][1:] for i in range(self.nquery, len(tokens))]
                # token_type.append(([0] * len(q) + [1] * (self.max_len - len(q)))[:self.max_len])
                padresult = [self.padding(inp) for inp in inps]
                inpids.append([res[0] for res in padresult])
                mask.append([res[1] for res in padresult])

        # for inp in inpids:
        #     for l in inp:
        #         print(self.tokenizer.decode(l))
        # print("==" * 20)
        model_inputs = {
            "input_ids": inpids,
            "attention_mask": mask,
            # "token_type_ids": token_type,
        }

        for key in model_inputs:
            model_inputs[key] = torch.LongTensor(model_inputs[key])

        return model_inputs