## declare an array variable
declare -a arr=("all" "random" "fisher")

## now loop through the above array
for i in "${arr[@]}"
do
if [ $i = all ]; then
    echo "The strings are equal."
else
    echo "The strings are not equal."
fi
done


exp_name=random
DATASET_PATH=/home/nmsl/nbv_simulator_data/
EXP_PATH=/home/nmsl/FisherRF/exp_results/
echo $DATASET_PATH

# mkdir ../nbv_plot_data/$exp_name
# mv ${EXP_PATH}/algo_time.csv ../nbv_plot_data/$exp_name/algo_time.csv
# mv ${EXP_PATH}/captured_time.csv ../nbv_plot_data/$exp_name/captured_time.csv
# mv ${EXP_PATH}/flying_time.csv ../nbv_plot_data/$exp_name/flying_time.csv
# mv ${EXP_PATH}/training_time.csv ../nbv_plot_data/$exp_name/training_time.csv


cp -r $DATASET_PATH/test1/transforms_test.json $DATASET_PATH/test2/transforms_test.json