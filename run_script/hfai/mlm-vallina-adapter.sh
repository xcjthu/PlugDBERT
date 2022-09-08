#!/bin/sh
source hfai_env xcj_env
python train.py -c config/hfai/VallinaAdapter.config 2>&1 | tee -a log/hfai/vallina-adapter.log
