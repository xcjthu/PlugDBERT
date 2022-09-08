from .Pretrain.MLMPretrain import MLMPretrain
from .Pretrain.VallinaPretrain import VallinaPretrain
model_list = {
    "mlm": MLMPretrain,
    "vallina": VallinaPretrain
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
