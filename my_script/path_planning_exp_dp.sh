#{random, fisher}
exp_name=dp_20v_30u_new
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
sampling_num=20
time_budget=100
training_time_limit=30

python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval \
        --time_budget $time_budget --training_time_limit $training_time_limit\
        --planner_type ours --sampling_method random --sampling_num $sampling_num --planning_method dp\
        --white_background --save_ply_after_last_adding --save_ply_each_time






