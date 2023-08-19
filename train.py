import argparse
import os
from threading import local
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train
from tools import print_rank
from tools.eval_tool import valid
from tools.init_tool import init_test_dataset, init_formatter
import random

import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_hyper_para(setting, config):
    if setting is None:
        return None
    pat = re.compile("\+(.*?)=(.*?)=(.*?)\+")
    paras = pat.findall(setting)
    for para in paras:
        print_rank("add", para)
        config.set(para[0], para[1], para[2])

def print_config(config):
    for sec in config.sections():
        print_rank("[%s]" % sec)
        for op in config.options(sec):
            print_rank("%s: %s" % (op, config.get(sec, op)))
        print_rank("========")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def get_args(local_rank=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    if local_rank is None:
        parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument('--only_eval', help="only do evaluation", action="store_true")
    parser.add_argument('--hyper_para', "-hp", default=None)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    if not local_rank is None:
        args.local_rank = local_rank

    print_rank("read config from", args.config)
    return args, create_config(args.config)

def main():
    args, config = get_args()
    set_seed(args.seed)


    gpu_list = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    print_rank(args.hyper_para)
    parse_hyper_para(args.hyper_para, config)

    # os.system("clear")
    config.set('distributed', 'local_rank', args.local_rank)

    torch.cuda.set_device(gpu_list[args.local_rank])
    torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
    config.set('distributed', 'gpu_num', len(gpu_list))

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, "train", local_rank = args.local_rank)
    do_test = False
    if args.do_test:
        do_test = True

    if args.local_rank <= 0:
        print_config(config)
    if not args.comment is None:
        print_rank(args.comment)
    if not args.only_eval:
        train(parameters, config, gpu_list, do_test, args.local_rank)
    with torch.no_grad():
        valid(parameters["model"], parameters["valid_dataset"], 0, config, gpu_list, parameters["output_function"])
        if do_test:
            init_formatter(config, ["test"])
            test_dataset = init_test_dataset(config)
            valid(parameters["model"], test_dataset, 0, config, gpu_list, parameters["output_function"], mode="test")


if __name__ == "__main__":
    print_rank(os.getcwd())
    main()

