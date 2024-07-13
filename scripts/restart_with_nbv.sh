DATASET_PATH=$1
EXP_PATH=$2
# experiment_id=$3
iteration_base=3000
M=5
epochs=0


for (( i=1; i<=$M; i++ ))
do
    python simulator_experiment_restart_with_nbv.py -s $DATASET_PATH -m $EXP_PATH \
            --eval --schema test --iterations $iteration_base --planner_type fisher \
            --iteration_base $iteration_base --num_init_views $i --interval_epochs $epochs --maximum_view $M --add_view 1 \
            --save_ply_each_time --white_background
    # cp $EXP_PATH/point_cloud/iteration_$((iteration_base * i))/point_cloud.ply $EXP_PATH/input.ply
    # cp $EXP_PATH/point_cloud/iteration_$((iteration_base * i))/point_cloud.ply $DATASET_PATH/fisher/1/points3d.ply
done