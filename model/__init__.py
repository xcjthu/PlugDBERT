from .Pretrain.MLMPretrain import MLMPretrain
from .Pretrain.VallinaPretrain import VallinaPretrain,VallinaDeltaPretrain
from .SQuAD.SQuAD import SQuAD
from .Fewrel.Fewrel import Fewrel
from .Pretrain.MultiMLMPretrain import MultiMLMPretrain

model_list = {
    "mlm": MLMPretrain,
    "vallina": VallinaPretrain,
    "vallina_delta": VallinaDeltaPretrain,
    "SQuAD": SQuAD,
    "Fewrel": Fewrel,
    "multi": MultiMLMPretrain,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
