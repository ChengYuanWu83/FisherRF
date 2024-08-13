#{random, sphere}
exp_name=$1
radius=$2
sampling_method=$3
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
sampling_num=50
time_budget=100
training_time_limit=30

python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval \
        --time_budget $time_budget  --training_time_limit $training_time_limit\
        --planner_type ours --sampling_method $sampling_method --sampling_num $sampling_num --planning_method a_star\
        --white_background --save_ply_after_last_adding --save_ply_each_time --radius_start $radius --radius_end $radius






