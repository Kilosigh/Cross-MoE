#!/bin/bash

# # 使用for循环打印从1到5的数字
# for i in {5..8}
# do
#   # echo "当前数字是: $i"
#   # ./run_baseline.sh 6 6 1 $i 2
#   ./run_all_ca.sh 6 6 0 $i 4
# done

<<<<<<< HEAD
# for i in {4..4}
# do
#   ./mix_type_2/run_all_10.sh 6 6 0 $((4*i))
# done

for i in {1..4}
do
  ./mix_type_2/run_all_10.sh 6 6 0 $((4*i))
=======
# for i in {1..4}
# do
#   ./mix_type_2/run_all_10.sh 6 6 0 $((4*i))
# done

# for i in {1..2}
# do
#   ./mix_type_2/run_all_11.sh 6 6 0 $((4*i))
# done

BALANCE_LOSS_WEIGHTS=("0.0001" "0.001" "0.005" "0.01" "0.1")

for weight in "${BALANCE_LOSS_WEIGHTS[@]}"; do
    ./mix_type_2/run_all_MoE_coeff.sh 6 6 0 8 $weight
>>>>>>> efc7024cd7e968dbacbcf4db525746db64e2d41a
done