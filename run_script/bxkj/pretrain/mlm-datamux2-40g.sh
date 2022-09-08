#!/bin/sh
python3 -m torch.distributed.launch --nproc_per_node=8 \
    train.py \
    -c config/bxkj/pretrain/wiki-webtext-mlm-mux2.config \
    -g 0,1,2,3,4,5,6,7