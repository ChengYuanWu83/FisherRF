#{random, fisher}
exp_name=random2
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
sampling_num=50
time_budget=100

python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval \
        --time_budget $time_budget \
        --planner_type random --sampling_method random --sampling_num $sampling_num \
        --white_background --save_ply_after_last_adding --save_ply_each_time






