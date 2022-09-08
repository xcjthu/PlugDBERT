#!/bin/sh
source hfai_env xcj_env
python train.py -c config/hfai/VallinaPile.config 2>&1 | tee -a log/hfai/vallina.log
