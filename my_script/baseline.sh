declare -a sampling_methods=("circular") #"random"
declare -a selecting_methods=("all")    #"fisher"
index=1
total_iteration=5000
time_budget=100

for sampling_method in "${sampling_methods[@]}"
do
    # start capture image
    DATASET_PATH=/home/nmsl/nbv_simulator_data/${sampling_method}_${time_budget}_${index}
    python ./train_with_plan/drone_capture.py --experiment_path $DATASET_PATH\
        --set_type train --time_budget $time_budget --sampling_method $sampling_method\
        --sort_view --record --radius_start 4 --radius_end 10 --views_num 50

    for selecting_method in "${selecting_methods[@]}"
    do 
        echo "$sampling_method/$selecting_method"
        EXP_PATH=/home/nmsl/FisherRF/exp_results/${sampling_method}_${selecting_method}_${time_budget}_${index}
        if [ $selecting_method = "all" ]; then
            python ./train_with_plan/train_extend.py -s $DATASET_PATH -m $EXP_PATH --iterations $total_iteration --white_background --eval
        elif [ $selecting_method = "fisher" ]; then
            python ./train_with_plan/active_train_bvs.py -s $DATASET_PATH -m $EXP_PATH\
            --iterations $total_iteration --eval --method=rand --schema test\
            --num_init_views 1 --interval_epochs 300 --maximum_view 3\
            --add_view 1 --iteration_base 0 --white_background 
        else
            echo "error"
        fi
    done
done
