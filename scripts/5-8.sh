#!/bin/bash

# # 使用for循环打印从1到5的数字
# for i in {5..8}
# do
#   # echo "当前数字是: $i"
#   # ./run_baseline.sh 6 6 1 $i 2
#   ./run_all_ca.sh 6 6 0 $i 4
# done

for i in {4..6}
do
  # echo "当前数字是: $i"
  # ./run_baseline.sh 6 6 1 $i 2
  ./run_w_o_text_2.sh $i $i 0
done