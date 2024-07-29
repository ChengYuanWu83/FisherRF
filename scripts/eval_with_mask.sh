DATASET_PATH=/home/nmsl/nbv_simulator_data
EXP_PATH=/home/nmsl/FisherRF/exp_results
declare -a arr=("change_algo/circular_new_test")
# declare -a arr=("random" "fisher")
# for i in "${arr[@]}"
# do
#    echo "$i"
#    # or do whatever with individual element of the array
# done
# python ./train_with_plan/random_capture.py --experiment_path $DATASET_PATH/test_set\
#         --set_type test --radius 4 --candidate_views_num 50

#eval
for i in "${arr[@]}"
do
        # cp -r $DATASET_PATH/test_set/test $DATASET_PATH/$i/test
        # cp $DATASET_PATH/test_set/transforms_test.json $DATASET_PATH/$i/transforms_test.json
        python render.py -s $DATASET_PATH/$i -m $EXP_PATH/$i --eval
        python generate_mask.py -m $EXP_PATH/$i
        python evaluation.py -m $EXP_PATH/$i
done