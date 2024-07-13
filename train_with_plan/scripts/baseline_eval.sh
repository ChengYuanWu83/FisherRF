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


#eval
for i in "${arr[@]}"
do
        python render.py -s $DATASET_PATH -m $EXP_PATH/$i --eval
        python metrics.py -m $EXP_PATH/$i
done