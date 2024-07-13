DATASET_PATH=$1
EXP_PATH=$2
experiment_id=$3

echo "run simulator"
source ~/rotorS_ws/devel/setup.bash
roslaunch rotors_gazebo mav_waypoint_with_path_planner.launch &
uav_simulator_pid=$!
sleep 5

roslaunch ORB_SLAM3 os3_mono.launch \
        vocabulary:=/home/nmsl/ORB_SLAM3/Vocabulary/ORBvoc.txt\
        cameraSetting:=/home/nmsl/ORB_SLAM3/Examples_old/Monocular/rotors.yaml \
        image:=/firefly/vi_sensor/left/image_raw \
        transform:=/firefly/odometry_sensor1/transform \
        sim:=$sim\
        algo:=$algo\
        traj:=$traj\
        scene:=$scene\
        wind:=$wind &
echo python simulator_experiment.py -s $DATASET_PATH/$experiment_id -m $EXP_PATH/$experiment_id --eval --schema score_test --iterations 7000  --white_background --planner_type fisher
python simulator_experiment.py -s $DATASET_PATH/$experiment_id -m $EXP_PATH/$experiment_id --eval --schema score_test --iterations 7000  --white_background --planner_type fisher --experiment_id $experiment_id


# end
kill $uav_simulator_pid

