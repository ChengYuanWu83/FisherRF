exp_name=ours
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
iteration_base=0
interval_epochs=500
M=30
epochs=0

python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval --schema test\
        --num_init_views 1 --interval_epochs $interval_epochs --maximum_view $M\
        --add_view 1 --iteration_base $iteration_base --planner_type $exp_name \
        --white_background --save_ply_after_last_adding
