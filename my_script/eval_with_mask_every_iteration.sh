DATASET_PATH=/home/nmsl/nbv_simulator_data
EXP_PATH=/home/nmsl/FisherRF/exp_results
declare -a index_set=(1)
declare -a time_constraints=(1)
declare -a time_budgets=(100)
declare -a sampling_methods=("circular")
declare -a sampling_numbers=(10 40 80)
declare -a algorithms=("dp" "astar")
total_iteration=30000
training_time_limit=30
#

for id in "${index_set[@]}"
do 
    for time_constraint in "${time_constraints[@]}"
    do  
        for time_budget in "${time_budgets[@]}"
        do 
            for sampling_method in "${sampling_methods[@]}"
            do  
                for sampling_number in "${sampling_numbers[@]}"
                do  
                    for algo in "${algorithms[@]}"
                    do  
                        i=${sampling_method}_${algo}_tb${time_budget}_sn${sampling_number}_id${id}_cabin
                        echo $i
                        if [ -d "$DATASET_PATH/$i/test" ]; then
                                rm -rf "$DATASET_PATH/$i/test"
                        fi
                        cp -r $DATASET_PATH/test_set_cabin/test $DATASET_PATH/$i/test
                        cp $DATASET_PATH/test_set/transforms_test.json $DATASET_PATH/$i/transforms_test.json

                        point_cloud_path=$(find $EXP_PATH/$i/point_cloud -name 'iteration_*')
                        for file in $point_cloud_path; do
                            number=$(echo "$file" | awk -F'iteration_' '{print $2}')                   
                            python render.py -s $DATASET_PATH/$i -m $EXP_PATH/$i --eval --iteration $number
                            
                        done
                        python generate_mask.py -m $EXP_PATH/$i
                        python evaluation.py -m $EXP_PATH/$i
                    done
                done
            done
        done
    done
done