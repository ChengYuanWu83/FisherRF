declare -a index_set=(1 2 3 4 5)
declare -a time_budgets=(100)
declare -a sampling_methods=("circular")
declare -a sampling_numbers=(20)
declare -a algorithms=("fisher" )
total_iteration=30000
training_time_limit=30
#

for id in "${index_set[@]}"
do 
    for time_budget in "${time_budgets[@]}"
    do 
        for sampling_method in "${sampling_methods[@]}"
        do  
            for sampling_number in "${sampling_numbers[@]}"
            do  
                for algo in "${algorithms[@]}"
                do  
                    exp_name=${sampling_method}_${algo}_tb${time_budget}_sn${sampling_number}_tc1_id${id}
                    DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
                    EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
                    
                    cp -r /home/nmsl/nbv_simulator_data/${sampling_method}_all_tb${time_budget}_sn${sampling_number}_tc1_id${id} $DATASET_PATH 

                    dp_folder=/home/nmsl/nbv_simulator_data/${sampling_method}_dp_tb${time_budget}_sn${sampling_number}_tc1_id${id}/train
                    astar_folder=/home/nmsl/nbv_simulator_data/${sampling_method}_astar_tb${time_budget}_sn${sampling_number}_tc1_id${id}/train

                    # 計算兩個資料夾中的檔案數量
                    dp_count=$(ls -1q "$dp_folder" | wc -l)
                    astar_count=$(ls -1q "$astar_folder" | wc -l)
                    all_count=$(ls -1q "$DATASET_PATH/train" | wc -l)

                    # 找出最大的檔案數量
                    MAX_FILE_COUNT=$((dp_count > astar_count ? dp_count : astar_count))
                    file_count=$((MAX_FILE_COUNT > all_count ? all_count : MAX_FILE_COUNT))

                    python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH\
                        --iterations $total_iteration --eval --method=H_reg --schema test\
                        --num_init_views 1 --interval_epochs 300 --maximum_view $file_count\
                        --add_view 1 --iteration_base 0 --white_background
                done
            done
        done
    done
done