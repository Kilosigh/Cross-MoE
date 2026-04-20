#!/bin/bash
sudo /usr/local/cuda/bin/ncu -f  -o  moe_phase1_report --set full -k "regex:.*batched_flash_decoding_mha_phase1.*"  \
 --import-source yes \
 /home/kilosigh/anaconda3/envs/MMT/bin/python ../test.py

# sudo -E /usr/local/cuda/bin/nsys profile 
