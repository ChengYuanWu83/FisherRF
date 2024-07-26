#{random, fisher}
exp_name=fisher2
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
iteration_base=0
interval_epochs=500
M=30
epochs=0

python time_budget_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval --schema test\
        --planner_type fisher \
        --white_background --save_ply_after_last_adding --save_ply_each_time






