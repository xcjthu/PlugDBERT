from transformers import BertTokenizer,AutoConfig,BertForMaskedLM,AutoModelForMaskedLM
import torch
from torch import nn
from model.metric import softmax_acc
from opendelta import AdapterModel,Visualization
from opendelta.auto_delta import AutoDeltaModel

class VallinaPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VallinaPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = AutoModelForMaskedLM.from_pretrained(self.plm)
        # self.model = BertForMaskedLM(self.plm_config)
        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )

        return {"loss": out["loss"], "acc_result": cal_loss(out["loss"], None, acc_result)}


class VallinaDeltaPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VallinaDeltaPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = AutoModelForMaskedLM.from_pretrained(self.plm)
        Visualization(self.model).structure_graph()

        delta_model = AdapterModel(backbone_model=self.model,
                bottleneck_dim=config.getint("train", "bottleneck_dim"),
                modified_modules=["[r]encoder.layer.(\d)+\.attention.output.LayerNorm",
                                "[r]encoder.layer.(\d)+\.output.LayerNorm"]
            )
        delta_model.freeze_module(set_state_dict=True, exclude=["deltas"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )

        return {"loss": out["loss"], "acc_result": cal_loss(out["loss"], None, acc_result)}


def cal_loss(mlm_loss, mse_loss, acc_result):
    if acc_result is None:
        acc_result = {"mlm": 0, "mse": 0, "step": 0}
    if acc_result["step"] > 2000:
        acc_result = {"mlm": 0, "mse": 0, "step": 0}
    acc_result["step"] += 1
    acc_result["mlm"] += mlm_loss.item()
    if mse_loss is not None:
        acc_result["mse"] += mse_loss.item()
    return acc_result
