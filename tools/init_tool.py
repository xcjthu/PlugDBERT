import logging
import torch

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn
from tools import output_log,print_rank
import os

logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}

    output_log(logger, "Begin to initialize dataset and formatter...")
    if mode == "train":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

    output_log(logger, "Begin to initialize models...")

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    step = 0

    if len(gpu_list) > 0:
        if params['local_rank'] < 0:
            model = model.cuda()
        else:
            model = model.to(gpu_list[params['local_rank']])
        try:
            # print_rank("before init DDP model")
            # print_rank(model.state_dict().keys())
            print(params['local_rank'])
            model = nn.parallel.DistributedDataParallel(model, device_ids = [params['local_rank']])#, find_unused_parameters = True)
            # print_rank("init DDP model")
            # print_rank(model.state_dict().keys())
            # print_rank("==" * 20)
            # print_rank(model.module.state_dict().keys())
            # print_rank("==" * 20)
            # from IPython import embed; embed()
        except Exception as e:
            print(e)
            output_log(logger, "No init_multi_gpu implemented in the model, use single gpu instead.", logging.WARNING)
    print_rank("=" * 10, "the keys in state dict", "=" * 10)
    print_rank(model.module.state_dict().keys())
    print_rank("==" * 15)

    try:
        if checkpoint is None and config.getboolean("output", "load_from_path"):
            path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
            files = os.listdir(path)
            print_rank("may load from", path, files)
            fid = max([int(f.split(".")[0]) for f in files])
            checkpoint = os.path.join(path, "%d.pkl" % fid)
        output_log(logger, "Begin to load checkpoint... from %s" % checkpoint)
        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"], strict=False)
        else:
            model.load_state_dict(parameters["model"])

        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                try:
                    optimizer.load_state_dict(parameters["optimizer"])
                except Exception as e:
                    output_log(logger, "ERROR do not load parameters of optimizer. %s" % e, logging.WARNING)
            else:
                output_log(logger, "Optimizer changed, do not load parameters of optimizer.", logging.WARNING)

            if "step" in parameters:
                step = parameters["step"]
            if "lr_scheduler" in parameters:
                result["lr_scheduler"] = parameters["lr_scheduler"]
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            output_log(logger, information, logging.ERROR)
            raise e
        else:
            output_log(logger, information, logging.WARNING)

    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["step"] = step

    output_log(logger, "Initialize done.", logging.INFO)

    return result
