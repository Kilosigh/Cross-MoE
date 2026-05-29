#!/bin/bash
sudo -E /usr/local/cuda/bin/nsys profile \
    --trace=cuda,osrt,nvtx \
    --trace-fork-before-exec=true \
    --cuda-memory-usage=true \
    --gpu-metrics-device=0 \
    --gpu-metrics-frequency=10000 \
    --output=moe_attention_report \
    --force-overwrite=true \
    --stats=true \
    /home/kilosigh/anaconda3/envs/MMT/bin/python ../split_k_fw_plus_bwd.py
