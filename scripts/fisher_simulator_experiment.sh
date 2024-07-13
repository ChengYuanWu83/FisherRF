DATASET_PATH=$1
EXP_PATH=$2
# experiment_id=$3
M=5
epochs=400

python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
            --eval --schema test --iterations 7000 --planner_type fisher \
            --iteration_base 2000 --num_init_views 1 --interval_epochs $epochs --maximum_view $M --add_view 1 \
            --save_ply_each_time --white_background
