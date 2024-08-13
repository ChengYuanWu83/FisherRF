declare -a radius_list=(9 10)
sampling_method=circular
time_budget=100

for radius in "${radius_list[@]}"
do  
    #
    exp_name=${sampling_method}_astar_tb${time_budget}_r${radius}
    bash ./my_script/path_planning_exp_a_star.sh $exp_name $radius $sampling_method

    # 
    DATASET_PATH=/home/nmsl/nbv_simulator_data/${sampling_method}_tb${time_budget}_r${radius}
    python ./train_with_plan/drone_capture.py --experiment_path $DATASET_PATH\
        --set_type train --time_budget $time_budget --sampling_method $sampling_method\
        --sort_view --record --radius_start $radius --radius_end $radius --views_num 50
        
    # 
    EXP_PATH=/home/nmsl/FisherRF/exp_results/${sampling_method}_all_tb${time_budget}_r${radius}
    python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH --iterations 30000 --white_background --eval
    # 
    folder_path=/home/nmsl/nbv_simulator_data/$exp_name/train
    file_count=$(find "$folder_path" -maxdepth 1 -type f | wc -l)

    EXP_PATH=/home/nmsl/FisherRF/exp_results/${sampling_method}_fisher_tb${time_budget}_r${radius}
    python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH\
        --iterations 30000 --eval --method=H_reg --schema test\
        --num_init_views 1 --interval_epochs 300 --maximum_view $file_count\
        --add_view 1 --iteration_base 0 --white_background --save_ply_each_time

done