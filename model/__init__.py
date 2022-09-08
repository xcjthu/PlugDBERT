from .Pretrain.MLMPretrain import MLMPretrain
from .Pretrain.VallinaPretrain import VallinaPretrain,VallinaDeltaPretrain

model_list = {
    "mlm": MLMPretrain,
    "vallina": VallinaPretrain,
    "vallina_delta": VallinaDeltaPretrain
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
