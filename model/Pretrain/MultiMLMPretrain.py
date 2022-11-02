from dis import dis
from transformers import BertTokenizer,AutoConfig,BertForMaskedLM,AutoModelForMaskedLM
import torch
from torch import nn,Tensor
from tools import print_rank
from model.metric import softmax_acc
from opendelta import AdapterModel,Visualization,LoraModel
from opendelta.auto_delta import AutoDeltaModel
from typing import OrderedDict
import torch.nn.functional as F
import torch.distributed as dist
import math

class MultiMLMPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MultiMLMPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = AutoModelForMaskedLM.from_pretrained(self.plm)
        if dist.get_rank() == 0:
            Visualization(self.model).structure_graph()

        self.domain_delta = AdapterModel(backbone_model=self.model,
                bottleneck_dim=config.getint("train", "bottleneck_dim"),
                modified_modules=["[r]encoder.layer.(\d)+\.attention.output.LayerNorm",
                                "[r]encoder.layer.(\d)+\.output.LayerNorm"]
            )

        self.domain_delta.freeze_module(set_state_dict=True, exclude=["deltas"])
        if dist.get_rank() == 0:
            self.domain_delta.log(delta_ratio=True, trainable_ratio=True, visualization=True)

        # print_rank("init delta model")

        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

        self.add_task_plugin(config.get("model", "task_plugin_path"))

    def add_task_plugin(self, path=None):
        if path is None:
            return
        print_rank("load task plugin from", path)
        self.task_delta = LoraModel(backbone_model=self.model,
                lora_r=32,
                lora_alpha=64,
                modified_modules=["self.query", "self.value"]
            )
        if dist.get_rank() == 0:
            self.task_delta.log(delta_ratio=True, trainable_ratio=True, visualization=True)
        ret = self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)["model"], False)
        # self.domain_delta.freeze_module(set_state_dict=True, exclude=["adapter"])
        for n, p in self.model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
        para_with_grad = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                para_with_grad.append(n)
        print_rank("=" * 10, "parameters with grad", "=" * 10)
        print_rank(para_with_grad)
        print_rank("==" * 20)

        print_rank("=" * 10, "load task plugin", "=" * 10)
        print_rank(ret)
        print_rank("==" * 20)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        print_rank("our self load function")
        ret = self.model.load_state_dict(state_dict, strict=strict)
        print_rank(ret)
        return ret

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        self.task_delta.detach()
        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )
        self.task_delta.attach()

        # cal distil loss
        out2 = self.model(
            input_ids=data["input_ids_ori"],
            attention_mask=data["attention_mask_ori"],
        )
        self.domain_delta.detach()
        old_out2 = self.model(
            input_ids=data["input_ids_ori"],
            attention_mask=data["attention_mask_ori"]
        )
        self.domain_delta.attach()

        _logits = out2["logits"]
        old_logits = old_out2["logits"]
        _targets = data["labels_ori"]
        pred = F.log_softmax(_logits[torch.where(_targets!=-100)], dim=-1)
        targ = F.softmax(old_logits[torch.where(_targets!=-100)],dim=-1)
        kl_loss = F.kl_div(pred, targ, reduction='none')
        _loss = kl_loss.sum(dim=1).mean()

        return {"loss": out["loss"] + _loss, "acc_result": cal_loss(out["loss"], _loss, acc_result)}


class MultiMLMFullPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MultiMLMFullPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = AutoModelForMaskedLM.from_pretrained(self.plm)

        self.teacher_model = AutoModelForMaskedLM.from_pretrained(self.plm)

        # print_rank("init delta model")

        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

        self.add_task_plugin(config.get("model", "task_plugin_path"))

    def add_task_plugin(self, path=None):
        if path is None:
            return
        print_rank("load task plugin from", path)
        self.task_delta = LoraModel(backbone_model=self.model,
                lora_r=32,
                lora_alpha=64,
                modified_modules=["self.query", "self.value"]
            )
        
        if dist.get_rank() == 0:
            self.task_delta.log(delta_ratio=True, trainable_ratio=True, visualization=True)
        ret = self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)["model"], False)
        
        # self.domain_delta.freeze_module(set_state_dict=True, exclude=["adapter"])
        for n, p in self.model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
        para_with_grad = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                para_with_grad.append(n)
        print_rank("=" * 10, "parameters with grad", "=" * 10)
        print_rank(para_with_grad)
        print_rank("==" * 20)

        print_rank("=" * 10, "load task plugin", "=" * 10)
        print_rank(ret)
        print_rank("==" * 20)

        self.teacher_delta = LoraModel(backbone_model=self.teacher_model,
                lora_r=32,
                lora_alpha=64,
                modified_modules=["self.query", "self.value"]
            )
        self.teacher_model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)["model"], False)
        for n, p in self.teacher_model.named_parameters():
            p.requires_grad = False

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        print_rank("our self load function")
        ret = self.model.load_state_dict(state_dict, strict=strict)
        print_rank(ret)
        return ret

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        self.task_delta.detach()
        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )
        self.task_delta.attach()

        # cal distil loss
        out2 = self.model(
            input_ids=data["input_ids_ori"],
            attention_mask=data["attention_mask_ori"],
        )

        with torch.inference_mode():
            old_out2 = self.teacher_model(
                input_ids=data["input_ids_ori"],
                attention_mask=data["attention_mask_ori"]
            )

        _logits = out2["logits"]
        old_logits = old_out2["logits"].clone()
        _targets = data["labels_ori"]
        pred = F.log_softmax(_logits[torch.where(_targets!=-100)], dim=-1)
        targ = F.softmax(old_logits[torch.where(_targets!=-100)],dim=-1)
        kl_loss = F.kl_div(pred, targ, reduction='none')
        _loss = kl_loss.sum(dim=1).mean()

        return {"loss": out["loss"] + _loss, "acc_result": cal_loss(out["loss"], _loss, acc_result)}



def cal_loss(mlm_loss, kd_loss, acc_result):
    if acc_result is None:
        acc_result = {"mlm": 0, "kdloss": 0, "step": 0}
    if acc_result["step"] > 2000:
        acc_result = {"mlm": 0, "kdloss": 0, "step": 0}
    acc_result["step"] += 1
    acc_result["mlm"] += mlm_loss.item()
    if kd_loss is not None:
        acc_result["kdloss"] += kd_loss.item()
    return acc_result
