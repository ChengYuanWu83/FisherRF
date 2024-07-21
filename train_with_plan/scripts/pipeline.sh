exp_name=$1
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
iteration_base=0
interval_epochs=500
M=16
epochs=0


for (( i=6; i<$M; i++ ))
do
    
        python ./train_with_plan/nbv_planner.py --experiment_path $DATASET_PATH --radius 3

        python ./train_with_plan/active_train_nbv.py -s $DATASET_PATH -m $EXP_PATH/$i\
        --iterations $interval_epochs --eval --method=H_reg --schema test\
        --num_init_views $i --interval_epochs $iteration_base --maximum_view $((i+1))\
        --add_view 1 --iteration_base $interval_epochs --white_background --save_ply_each_time
done
python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH/final --iterations 30000 --white_background --eval

