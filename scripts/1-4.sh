#!/bin/bash

# 使用for循环打印从1到5的数字
# for i in {1..4}
# do
#   ./mix_type_2/run_all_00.sh 6 6 0 $((4*i))
# done

# for i in {1..4}
# do
#   ./mix_type_2/run_all_01.sh 6 6 0 $((4*i))
# done

for i in {4..4}
do
  ./mix_type_2/run_all_11.sh 6 6 0 $((4*i))
done

# for i in {1..4}
# do
#   # echo "当前数字是: $i"
#   # ./run_baseline.sh 6 6 0 $i 1
#   ./run_baseline_no_text.sh 6 6 0 $i 5
# done

