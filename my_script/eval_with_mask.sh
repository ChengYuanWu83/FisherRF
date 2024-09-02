DATASET_PATH=/home/nmsl/nbv_simulator_data/
EXP_PATH=/home/nmsl/FisherRF/exp_results/
scene=car

declare -a index_set=(1)
# 0.08: 4s, 0.04: 2s,0.02:1s, 0.01:0.5s, 0.005:0.25s
declare -a time_constraints=(0.2 0.002 0.0002) #0.08 0.04 0.02 0.01 0.005 
# declare -a scheduling_num_set=(3)
declare -a scheduling_windows=(50) #25 50 75 100
declare -a sampling_methods=("circular") 
declare -a sampling_numbers=(10) #5 10 20 40 80
declare -a algorithms=("dp") # "astar" "dp" "all" "fisher" 


total_iteration=10000
initial_training_time=1
#

for id in "${index_set[@]}"
do 
    for time_constraint in "${time_constraints[@]}"
    do  
        for scheduling_num in "${scheduling_num_set[@]}"
        do 
            for scheduling_window in "${scheduling_windows[@]}"
            do 
                for sampling_method in "${sampling_methods[@]}"
                do  
                    for sampling_number in "${sampling_numbers[@]}"
                    do  
                        for algo in "${algorithms[@]}"
                        do  
                            # exp_name=${sampling_method}_${algo}_tb${time_budget}_sn${sampling_number}_id${id}
                            i=${sampling_method}_${algo}_N${scheduling_num}_T${scheduling_window}_tf${time_constraint}_sn${sampling_number}_id${id}
                            echo $i
                            if [ -d "$DATASET_PATH/$i/test" ]; then
                                    rm -rf "$DATASET_PATH/$i/test"
                            fi
                            cp -r $DATASET_PATH/test_set_$scene/test $DATASET_PATH/$i/test
                            cp $DATASET_PATH/test_set_$scene/transforms_test.json $DATASET_PATH/$i/transforms_test.json
                            # test_set_cabin
                            python render.py -s $DATASET_PATH/$i -m $EXP_PATH/$i --eval
                            python generate_mask.py -m $EXP_PATH/$i
                            python evaluation.py -m $EXP_PATH/$i
                        done
                    done
                done
            done
        done
    done
done