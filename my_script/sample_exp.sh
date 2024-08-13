declare -a sampling_numbers=(60 70 80 90)
declare -a index_set=(1 2 3 4 5)
sampling_method=random
time_budget=100
total_iteration=30000
training_time_limit=30

for id in  "${index_set[@]}"
do 
    for sampling_num in "${sampling_numbers[@]}"
    do  
        #
        exp_name=${sampling_method}_dp_tb${time_budget}_sn${sampling_num}_id${id}
        DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
        EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
        python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
            --iterations $total_iteration --eval \
            --time_budget $time_budget  --training_time_limit $training_time_limit\
            --planner_type ours --sampling_method $sampling_method --sampling_num $sampling_num --planning_method dp\
            --white_background --save_ply_after_last_adding --save_ply_each_time --radius_start 4 --radius_end 10 
        #
        exp_name=${sampling_method}_astar_tb${time_budget}_sn${sampling_num}_id${id}
        DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
        EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
        python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
            --iterations $total_iteration --eval \
            --time_budget $time_budget  --training_time_limit $training_time_limit\
            --planner_type ours --sampling_method $sampling_method --sampling_num $sampling_num --planning_method a_star\
            --white_background --save_ply_after_last_adding --save_ply_each_time --radius_start 4 --radius_end 10
    done
done