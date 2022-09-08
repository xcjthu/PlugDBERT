#!/bin/sh
python3 -m torch.distributed.launch --nproc_per_node=8 \
    train.py \
    -c config/bxkj/pretrain/wiki-webtext-mlm.config \
    -g 0,1,2,3,4,5,6,7 
    #--checkpoint /data/home/scv0540/xcj/datamux/checkpoint/mlm-webtext/2.pkl