from transformers import BertTokenizer,AutoConfig,RobertaModel,RobertaConfig
import torch
from torch import nn,Tensor
from model.metric import softmax_acc
from opendelta import AdapterModel,Visualization,LoraModel
from opendelta.auto_delta import AutoDeltaModel
import os
from typing import OrderedDict
from tools import print_rank

class Fewrel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Fewrel, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = RobertaModel.from_pretrained(self.plm)
        Visualization(self.model).structure_graph()

        delta_model = LoraModel(backbone_model=self.model,
                lora_r=config.getint("train", "lora_r"),
                lora_alpha=config.getint("train", "lora_alpha"),
                modified_modules=["self.query", "self.value"]
            )
        delta_model.freeze_module(set_state_dict=True, exclude=["deltas"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
        self.output = nn.Linear(self.plm_config.hidden_size, 1)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

        self.nshots = config.getint("train", "nshot")
        self.nways = config.getint("train", "nways")

        self.add_domain_plugin(config.get("model", "domain_plugin_path"))
    
    def add_domain_plugin(self, path=None):
        if path is None:
            return
        print_rank("load domain plugin from", path)
        domain_delta = AdapterModel(backbone_model=self.model,
                bottleneck_dim=64,
                modified_modules=["[r]encoder.layer.(\d)+\.attention.output.LayerNorm",
                                "[r]encoder.layer.(\d)+\.output.LayerNorm"]
            )
        domain_delta.log(delta_ratio=True, trainable_ratio=True, visualization=True)
        self.model.load_state_dict(torch.load(path)["model"], False)


    def state_dict(self):
        ret = self.model.state_dict()
        linear_state = self.output.state_dict()
        for key in linear_state:
            ret[f"linear-output.{key}"] = linear_state[key]
        return ret

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        print("our self load function")
        linear_state = OrderedDict()
        self.output.load_state_dict({
            "weight": state_dict["linear-output.weight"],
            "bias": state_dict["linear-output.bias"],
        })
        # for key in state_dict:
        #     if key[:13] == "linear-output":
        #         linear_state[key[14:]] = state_dict.pop(key)
        # self.output.load_state_dict(linear_state)
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, data, config, gpu_list, acc_result, mode):
        batch, ins_num, seq_len = data["input_ids"].size()
        input_ids, attention_mask = data["input_ids"].view(batch * ins_num, seq_len),  data["attention_mask"].view(batch * ins_num, seq_len)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=data["token_type_ids"]
        )
        score = self.output(out["pooler_output"]).unsqueeze(-1)

        score = score.view(batch, self.nways, self.nshots).mean(dim=-1)
        labels = torch.zeros(batch, dtype=torch.long, device=data["input_ids"].device)
        loss = self.loss_func(score, labels)
        acc_result = softmax_acc(score, labels, acc_result)

        return {"loss": loss, "acc_result": acc_result}
