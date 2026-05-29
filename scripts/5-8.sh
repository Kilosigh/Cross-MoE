#!/bin/bash

# # 使用for循环打印从1到5的数字
# for i in {5..8}
# do
#   # echo "当前数字是: $i"
#   # ./run_baseline.sh 6 6 1 $i 2
#   ./run_all_ca.sh 6 6 0 $i 4
# done

# for i in {1..4}
# do
#   ./mix_type_2/run_all_10.sh 6 6 0 $((4*i))
# done

# for i in {1..2}
# do
#   ./mix_type_2/run_all_11.sh 6 6 0 $((4*i))
# done

# BALANCE_LOSS_WEIGHTS=( "0.01"  "0.05" "0.1")
# # BALANCE_LOSS_WEIGHTS=("0.1")

# for weight in "${BALANCE_LOSS_WEIGHTS[@]}"; do
#     ./mix_type_2/run_all_MoE_coeff_2.sh 6 6 0 8 $weight
# done


# for i in {3..4}
# do
#   # echo "当前数字是: $i"
#   # ./run_baseline.sh 6 6 0 $i 1
#   ./traditional_dataset/run_all_uni_2.sh 6 6 0 $((4*i))
# done


for i in {2..4}
do
  # echo "当前数字是: $i"
  # ./run_baseline.sh 6 6 0 $i 1
  # ./traditional_dataset/run_all_uni_2.sh 3 3 1 $((4*i))
  ./traditional_dataset/run_all_01_3.sh 3 3 1 $((4*i))
done