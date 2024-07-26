declare -a arr=("random" "fisher")
total_iteration=30000
iteration_base=0
interval_epochs=500
M=10
epochs=0

        
## now loop through the above array
for i in "${arr[@]}"
do
        echo "run simulator "
        source ~/rotorS_ws/devel/setup.bash
        roslaunch rotors_gazebo mav_hovering_example_with_unity.launch & 
        simulator_pid=$!
        sleep 10

        DATASET_PATH=/home/nmsl/nbv_simulator_data/$i
        EXP_PATH=/home/nmsl/FisherRF/exp_results/$i

        echo "start training"
        python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} \
                --iterations $total_iteration --eval --schema test\
                --num_init_views 1 --interval_epochs $interval_epochs --maximum_view $M\
                --add_view 1 --iteration_base $iteration_base --planner_type $i \
                --white_background --save_ply_after_last_adding

        kill $simulator_pid
        sleep 10
done






