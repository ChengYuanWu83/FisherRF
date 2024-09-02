declare -a index_set=(1)
# 0.08: 4s, 0.04: 2s,0.02:1s, 0.01:0.5s, 0.005:0.25s
declare -a time_constraints=(0.02) #0.08 0.04 0.02 0.01 0.005 
# declare -a scheduling_num_set=(3)
declare -a scheduling_windows=(50) #25 50 75 100
declare -a sampling_methods=("circular") 
declare -a sampling_numbers=(10) #5 10 20 40 80
declare -a algorithms=("fisher") # "astar" "dp" "all" "fisher" 

deg="5deg"
scene=car
total_iteration=15000
baseline_iteration=30000
initial_training_time=1
total_budget=150
capture_budget=300
training_budget=50
TEST_DATASET_PATH=/home/nmsl/nbv_simulator_data/

#

for id in "${index_set[@]}"
do 
    for time_constraint in "${time_constraints[@]}"
    do  
        for scheduling_window in "${scheduling_windows[@]}"
        do 
            for sampling_method in "${sampling_methods[@]}"
            do  
                for sampling_number in "${sampling_numbers[@]}"
                do  
                    for algo in "${algorithms[@]}"
                    do  
                        # exp_name=${sampling_method}_${algo}_tb${time_budget}_sn${sampling_number}_id${id}
                        exp_name=${sampling_method}_${algo}_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_${scene}_id${id}
                        DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
                        EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
                        cp -r /home/nmsl/nbv_simulator_data/${sampling_method}_all_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_${scene}_id1 $DATASET_PATH 
                        echo "$exp_name"
                        if [ $algo == "all" ]; then
                            python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH --iterations $baseline_iteration \
                                --white_background --eval --time_budget $training_budget --record_gpu
                        elif [ $algo == "fisher" ]; then        
                            cp -r /home/nmsl/nbv_simulator_data/${sampling_method}_all_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_${scene}_id1 $DATASET_PATH 

                            dp_folder=/home/nmsl/nbv_simulator_data/${sampling_method}_dp_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_${scene}_id${id}/train
                            astar_folder=/home/nmsl/nbv_simulator_data/${sampling_method}_astar_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_${scene}_id${id}/train

                            # 計算兩個資料夾中的檔案數量
                            dp_count=$(ls -1q "$dp_folder" | wc -l)
                            astar_count=$(ls -1q "$astar_folder" | wc -l)
                            all_count=$(ls -1q "$DATASET_PATH/train" | wc -l)

                            # 找出最大的檔案數量
                            MAX_FILE_COUNT=$((dp_count > astar_count ? dp_count : astar_count))
                            MIN_FILE_COUNT=$((dp_count < astar_count ? dp_count : astar_count))
                            file_count=$((MAX_FILE_COUNT > all_count ? all_count : MAX_FILE_COUNT))
                            # file_count=$((MIN_FILE_COUNT < all_count ? MIN_FILE_COUNT : all_count))

                            python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH\
                                --iterations $baseline_iteration --eval --method=H_reg --schema test\
                                --num_init_views 1 --interval_epochs 300 --maximum_view $file_count\
                                --add_view 1 --iteration_base 0 --white_background --time_budget $training_budget
                        fi
                        # eval
                        if [ -d "$DATASET_PATH/test" ]; then
                            rm -rf "$DATASET_PATH/test"
                        fi
                        cp -r $TEST_DATASET_PATH/test_set_$scene/test $DATASET_PATH/test
                        cp $TEST_DATASET_PATH/test_set_$scene/transforms_test.json $DATASET_PATH/transforms_test.json
                        # test_set_cabin
                        python render.py -s $DATASET_PATH -m $EXP_PATH --eval
                        python generate_mask.py -m $EXP_PATH
                        python evaluation.py -m $EXP_PATH
                    done
                done
            done
        done
    done
done