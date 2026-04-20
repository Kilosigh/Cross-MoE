#!/bin/bash

# 使用for循环打印从1到5的数字

# for i in {1..4}
# do
#   ./mix_type_2/run_all_01.sh 6 6 0 $((4*i))
# done

for i in {1..4}
do
  # echo "当前数字是: $i"
  # ./run_baseline.sh 6 6 0 $i 1
  ./traditional_dataset/run_all_uni.sh 2 2 0 $((4*i))
done

# BALANCE_LOSS_WEIGHTS=("0.0001" "0.001" "0.005"  "0.01"  "0.05" "0.1")

# BALANCE_LOSS_WEIGHTS=("0.0001" "0.001" "0.005" )
# # BALANCE_LOSS_WEIGHTS=("0.005" )

# for weight in "${BALANCE_LOSS_WEIGHTS[@]}"; do
#     ./mix_type_2/run_all_MoE_coeff.sh 6 6 0 8 $weight
# done