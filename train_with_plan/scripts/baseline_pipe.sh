exp_name=baseline
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
# experiment_id=$3
total_iteration=30000
iteration_base=0
interval_epochs=500
M=16
epochs=0

## declare an array variable
declare -a arr=("all" "random" "fisher")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   # or do whatever with individual element of the array
done

# collect dataset
python ./train_with_plan/circular_capture.py --experiment_path $DATASET_PATH --radius 3 --views_num 30
python ./train_with_plan/random_capture.py --experiment_path $DATASET_PATH\
        --set_type test --radius 3 --candidate_views_num 50
for i in "${arr[@]}"
do
        if [ $i = "all" ]; then
                echo "Train all the images."
                python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH/$i\
                        --iterations $total_iteration --white_background --eval
        elif [ $i = "random" ]; then
                echo "Train the images which selected by random."
                python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH/$i\
                        --iterations $total_iteration --eval --method=rand --schema test\
                        --num_init_views 1 --interval_epochs $interval_epochs --maximum_view $M\
                        --add_view 1 --iteration_base $iteration_base --white_background 
        elif [ $i = "fisher" ]; then
                python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH/$i\
                        --iterations $total_iteration --eval --method=H_reg --schema test\
                        --num_init_views 1 --interval_epochs $interval_epochs --maximum_view $M\
                        --add_view 1 --iteration_base $iteration_base --white_background 
        else
                echo "unknown methods"
        fi
done

# #eval
# for i in "${arr[@]}"
# do
#         python render.py -s $DATASET_PATH -m $EXP_PATH/$i --eval
#         python metrics.py -m $EXP_PATH/$i
# done