import logging

try:
    import hfai
    import hfai.distributed as dist
    import hfai_env
    # hfai_env.set_env('xcj_env')
    _use_hfai=True
except:
    import torch.distributed as dist
    _use_hfai=False

def use_hfai():
    return _use_hfai


def output_log(logger: logging.Logger, info: str, level: int = logging.INFO, *args):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        logger._log(level, info, args)

def print_rank(*arg):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        print(*arg)

def get_world_size():
    return dist.get_world_size()

def get_rank():
    return dist.get_rank()

