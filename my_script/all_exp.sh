bash ./all_exp4000.sh
declare -a index_set=(1)
declare -a time_constraints=(1)
declare -a time_budgets=(25 50 200)
declare -a sampling_methods=("random")
declare -a sampling_numbers=(20)
declare -a algorithms=("astar" )


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
                        exp_name=${sampling_method}_${algo}_tb${time_budget}_sn${sampling_number}_id${id}
                        DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
                        EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name
                        echo "$exp_name"
                        if [ $algo == "dp" ]; then
                            python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
                                --iterations $total_iteration --eval \
                                --time_budget $time_budget  --training_time_limit $training_time_limit\
                                --planner_type ours --sampling_method $sampling_method --sampling_num $sampling_number --planning_method dp\
                                --white_background \
                                --radius_start 4 --radius_end 10 --time_constraint $time_constraint
                                # --white_background --save_ply_after_last_adding --save_ply_each_time \
                        elif [ $algo == "astar" ]; then
                            python path_planning_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
                                --iterations $total_iteration --eval \
                                --time_budget $time_budget  --training_time_limit $training_time_limit\
                                --planner_type ours --sampling_method $sampling_method --sampling_num $sampling_number --planning_method a_star\
                                --white_background \
                                --radius_start 4 --radius_end 10 --time_constraint $time_constraint
                                # --white_background --save_ply_after_last_adding --save_ply_each_time \
                        elif [ $algo == "all" ]; then
                            python ./train_with_plan/drone_capture.py --experiment_path $DATASET_PATH\
                                --set_type train --time_budget $time_budget --sampling_method $sampling_method\
                                --record --radius_start 4 --radius_end 10 --views_num $sampling_number --sort_view
                            python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH --iterations $total_iteration \
                                --white_background --eval 
                        fi
                    done
                done
            done
        done
    done
done