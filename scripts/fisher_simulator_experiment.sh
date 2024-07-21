DATASET_PATH=$1
EXP_PATH=$2
# experiment_id=$3
total_iteration=30000
iteration_base=0
interval_epochs=500
M=10
epochs=0

python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
        --iterations $total_iteration --eval --schema test\
        --num_init_views 1 --interval_epochs $interval_epochs --maximum_view $M\
        --add_view 1 --iteration_base $iteration_base --white_background --save_ply_each_time