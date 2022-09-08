#!/bin/sh
python3 -m torch.distributed.launch --nproc_per_node=8 \
    train.py \
    -c config/a40-1/wiki-webtext-mlm.config \
    -g 0,1,2,3,4,5,6,7 2>&1 | tee -a logs/a40/wiki-webtext-datamux8.log