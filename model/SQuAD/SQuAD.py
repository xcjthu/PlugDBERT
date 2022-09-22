from transformers import BertTokenizer,AutoConfig,BertForQuestionAnswering,RobertaForQuestionAnswering
import torch
from torch import nn
from model.metric import softmax_acc
from opendelta import AdapterModel,Visualization,LoraModel
from opendelta.auto_delta import AutoDeltaModel
from .utils_qa import *

class SQuAD(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SQuAD, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = AutoConfig.from_pretrained(self.plm)
        self.model = RobertaForQuestionAnswering.from_pretrained(self.plm)
        Visualization(self.model).structure_graph()

        delta_model = LoraModel(backbone_model=self.model,
                lora_r=config.getint("train", "lora_r"),
                lora_alpha=config.getint("train", "lora_alpha"),
                modified_modules=["self.query", "self.value"]
            )
        delta_model.freeze_module(set_state_dict=True, exclude=["deltas","qa_outputs"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

    def forward(self, data, config, gpu_list, acc_result, mode):
        # print(data.keys())
        if mode == "train":
            out = self.model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                # token_type_ids=data["token_type_ids"],
                start_positions=data["start_positions"],
                end_positions=data["end_positions"],
            )
            acc_result = start_end_acc(data["start_positions"], out["start_logits"], acc_result)
            acc_result = start_end_acc(data["end_positions"], out["end_logits"], acc_result)
            return {"loss": out["loss"], "acc_result": acc_result}
        else:
            out = self.model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                # token_type_ids=data["token_type_ids"],
            )
            predict = postprocess_qa_predictions(data["oridata"], data, out["start_logits"], out["end_logits"])
            return {"loss": 0, "acc_result": squad_metric(predict, data["id2ans"], acc_result)}


