#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=4 \
    train.py \
    -c config/SQuAD/LoRa.config \
    -g 0,1,2,3 \
    2>&1 | tee log/117/squad-lora.log