#!/bin/bash
sudo -E /usr/local/cuda/bin/nsys profile \
    -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o moe_mha_macro \
    --force-overwrite true \
    /home/kilosigh/anaconda3/envs/MMT/bin/python ../test.py
