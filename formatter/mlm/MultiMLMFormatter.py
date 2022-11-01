import json
from click import option
import torch
import os
import numpy as np

import random
from transformers import BertTokenizer,AutoTokenizer
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForLanguageModeling
from tools import print_rank

class MultiMLMFormatter:
    def __init__(self, config, mode, *args, **params):
        self.doc_len = config.getint("train", "doc_len")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.mode = mode
        self.config = config
        # print("formatter", config.get("model", "pretrained_model"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model", "pretrained_model"))
        self.masker = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_prob, return_tensors="pt")

    def process(self, data):
        # print_rank(data[0])
        ctxs = [d[0] for d in data]
        ctxs_ori = [d[1]["text"] for d in data]

        ctx_tokens = self.tokenizer(ctxs, max_length=self.doc_len, padding="max_length", truncation=True, return_tensors="pt")
        qinp, qlabels = self.masker.torch_mask_tokens(ctx_tokens["input_ids"])

        ret = {
            "input_ids": qinp,
            "attention_mask": ctx_tokens["attention_mask"],
            "labels": qlabels
        }

        ctx_tokens_ori = self.tokenizer(ctxs_ori, max_length=self.doc_len, padding="max_length", truncation=True, return_tensors="pt")
        qinp_ori, qlabels_ori = self.masker.torch_mask_tokens(ctx_tokens_ori["input_ids"])
        ret.update({
            "input_ids_ori": qinp_ori,
            "attention_mask_ori": ctx_tokens_ori["attention_mask"],
            "labels_ori": qlabels_ori,
        })

        return ret