#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=8 \
    train.py \
    -c config/pre-train/adapter.config \
    -g 0,1,2,3,4,5,6,7 2>&1 | tee -a log/117/adapter-pretrain.log