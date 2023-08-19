from transformers import BertTokenizer,AutoConfig,BertForMaskedLM,AutoModelForMaskedLM
import torch
from torch import nn,Tensor
from typing import OrderedDict
from model.PlugD.BERTPlugDHF import BERTPlugD

class PlugDPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PlugDPretrain, self).__init__()
        
        self.model = BERTPlugD(t5path=config.get("model", "pretrained_model"))
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, data, config, gpu_list, acc_result, mode):
        out = self.model(
            query_input_ids = data["query_input_ids"],
            query_attention_mask = data["query_mask"],
            ctx_input_ids = data["ctx_input_ids"],
            ctx_attention_mask = data["ctx_attention_mask"]
        )

        logits = out#.logits
        vocab_size = logits.size(-1)
        loss = self.loss_func(logits.view(-1, vocab_size), data["query_labels"].view(-1))
        acc_result = softmax_acc(logits.view(-1, vocab_size), data["query_labels"].view(-1), acc_result)
        if "mask" not in acc_result:
            acc_result["mask"] = 0
        acc_result["mask"] += int((data["query_input_ids"] != 103).sum())
        return {"loss": loss, "acc_result": acc_result}

def softmax_acc(logits, target, acc_result):
    if acc_result is None:
        acc_result = {"right": 0, "total": 0}
    predict = torch.argmax(logits, dim=-1)
    acc_result["right"] += int((predict == target).sum())
    acc_result["total"] += int((target != -100).sum())
    
    return acc_result
