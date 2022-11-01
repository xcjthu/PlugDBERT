from dis import dis
from transformers import BertTokenizer,AutoConfig,BertForMaskedLM,AutoModelForMaskedLM
import torch
from torch import nn,Tensor
from tools import print_rank
from model.metric import softmax_acc
from opendelta import AdapterModel,Visualization
from opendelta.auto_delta import AutoDeltaModel
from typing import OrderedDict
import torch.nn.functional as F
import torch.distributed as dist

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

        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        print_rank("our self load function")
        ret = self.model.load_state_dict(state_dict, strict=strict)
        print_rank(ret)
        return ret

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )

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
