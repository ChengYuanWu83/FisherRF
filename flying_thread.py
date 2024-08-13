import numpy as np
from planner.utils import sphere_sampling, view_to_pose_with_target_point
from scipy.spatial.transform import Rotation as R
import rospy
import time
import yaml
import os
from planner import get_planner
from argparse import ArgumentParser, Namespace
import threading

lock = threading.Lock()

def flying_uav(traj, time_budget, nbv_planner,event):
    step = 0 
    pose_idxs = 0
    while time_budget > 0:
        flying_start_time = time.time()
        nbv = traj[step]
        nbv_planner.move_sensor(nbv) #need to change
        step +=1
        flying_end_time = time.time()

        #[cyw]:captured time
        captured_start_time = time.time()
        nbv_planner.store_train_set()
        captured_end_time = time.time()
        pose_idxs += 1

        time_budget -=  (captured_end_time - flying_start_time)
        print(f"time_budget:{time_budget}")
        event.set()

def main():
    rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--experiment_path", type=str, default="/home/nmsl/nbv_simulator_data/not defined", help="must be defined in evaluation mode")
    parser.add_argument("--planner_type", "-P", type=str, default="random", help="planner_type")
    parser.add_argument("--time_budget", type=float, default=100.0, help="time budget")
    parser.add_argument("--training_time_limit", type=float, default=5.0, help="training_time_limit")
    args = parser.parse_args()

    experiment_path = args.experiment_path
    planner_type = args.planner_type
    time_budget = args.time_budget
    training_time_limit = args.training_time_limit

    print(
        f"---------- random planner ----------\n"
    )
    planner_cfg_path = os.path.join(
        "planner/config", f"random_planner.yaml"
    )
    print(planner_cfg_path)
    assert os.path.exists(planner_cfg_path)
    with open(planner_cfg_path, "r") as config_file:
        planner_cfg = yaml.safe_load(config_file)
    # planner_cfg.update(__dict__)
    planner_cfg["planner_type"] = planner_type
    planner_cfg["experiment_path"] = experiment_path
    print(planner_cfg)
    nbv_planner = get_planner(planner_cfg)
    camera_info = nbv_planner.simulator_bridge.camera_info


    traj = nbv_planner.plan_path() #first traj
    event = threading.Event() 

    flying_thread = threading.Thread(target = flying_uav, args = (traj, time_budget, nbv_planner,event))
    flying_thread.start()

    iteration = 0
    while iteration <  30000:
        if event.wait(0):
            print("update training set")
            event.clear() 
        print("training")
        iteration += 0.1
        time.sleep(1)

    flying_thread.join()

    print("DONE.")

if __name__ == "__main__":
    main()