# ## declare an array variable
declare -a arr=(4 10)
echo $arr
# # ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i/"
# done



#!/bin/bash
# DATASET_PATH=/home/nmsl/nbv_simulator_data
# i=circular_fisher_tb400_sn20_id3
# TT=$(find /home/nmsl/FisherRF/exp_results/circular_dp_tb100_sn20_tc1_id0/point_cloud -name 'iteration_*')

# 使用 echo 和 awk 仅提取 iteration_ 后面的数字
for file in $TT; do
    number=$(echo "$file" | awk -F'iteration_' '{print $2}')
    echo "$number"
done

# find "$BASE_DIR" -type d -name 'test' | while read TEST_DIR; do
#     # 在 test 子資料夾內查找符合 our_* 的檔案
#     FILE=$(find "$TEST_DIR" -maxdepth 1 -type f -name 'our_*')
    
#     # 如果檔案存在，進行處理
#     if [[ -f "$FILE" ]]; then
#         echo "Processing file: $FILE"
#         # 例如，讀取檔案內容
#         cat "$FILE"
#     fi
# done