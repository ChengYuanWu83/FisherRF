
import rospy
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from argparse import ArgumentParser
from planner import get_planner
from planner.utils import uniform_sampling, xyz_to_view, random_view, rotation_2_quaternion, sphere_sampling, sphere_sampling_unorder
import numpy as np
import yaml
import time
from planner import utils
import json
import imageio.v2 as imageio
import csv
from scipy.spatial.transform import Rotation as R

def setup_csv(path):

    flying_time_csv = f"{path}/flying_time.csv"
    flying_file_exists = os.path.isfile(flying_time_csv)
    if not flying_file_exists:
        with open(flying_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pose_idxs', 'times'])

    captured_time_csv = f"{path}/captured_time.csv"
    captured_file_exists = os.path.isfile(captured_time_csv)
    if not captured_file_exists:
        with open(captured_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pose_idxs', 'times'])

    return flying_time_csv, captured_time_csv

def create_empty_test_set(source_path, camera_info):
    test_path = os.path.join(source_path, "test")
    os.makedirs(test_path, exist_ok=True)

    camera_dict = utils.get_camera_json(camera_info) # camera_info json
    transform_json_file = (f"{source_path}/transforms_test.json")
    if not os.path.isfile(transform_json_file):
        with open(transform_json_file, "w") as f:
            json.dump(camera_dict, f, indent=4)  
    
    
def save_image_into_train_set(source_path, captured_pose, image, camera_info):
    print("------ record simulator data ------\n")
    # [cyw]:save transform_train.json
    camera_dict = utils.get_camera_json(camera_info) # camera_info json
    transform_json_file = (f"{source_path}/transforms_train.json")
    if os.path.isfile(transform_json_file):
        # Read the existing data
        with open(transform_json_file, 'r') as f:
            record_dict = json.load(f)
    else:
        with open(transform_json_file, "w") as f:
            json.dump(camera_dict, f, indent=4)
            record_dict = camera_dict

    i = len(record_dict["frames"])
    # transformation = np.array(
    # [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    # )  # transform gazebo coordinate to opengl format
    # transform = utils.quaternion_to_rotation_matrix(captured_pose)
    # # print(transform)
    # opengltransform = transform @ transformation
    
    # [cyw]:rotate to opengl transform
    transform = utils.quaternion_to_rotation_matrix(captured_pose)
    opengltransform = np.eye(4)
    euler = R.from_matrix(transform[:3, :3]).as_euler('xyz')
    # print(f"origin euler: roll: {euler[0]}, pitch: {euler[1]}, yaw:{euler[2]}")
    new_euler = np.empty(3)
    new_euler[0] = euler[1] + (np.pi/2)
    new_euler[1] = euler[0]
    new_euler[2] = euler[2] - (np.pi/2)
    opengl_rotation =  R.from_euler('xyz', [new_euler[0] , new_euler[1], new_euler[2]]).as_matrix()
    opengltransform[:3, -1] = transform[:3, -1]
    opengltransform[:3, :3] = opengl_rotation

    train_image_file = (f"./train/r_{i:04d}")
    data_frame = {
        "file_path": train_image_file,
        "transform_matrix": opengltransform.tolist(), #[cyw]: see waht happend if don't use it
    }
    record_dict["frames"].append(data_frame)
    with open(transform_json_file, "w") as f:
        json.dump(record_dict, f, indent=4)
    #[cyw]: save the image
    train_path = os.path.join(source_path, "train")
    os.makedirs(train_path, exist_ok=True)
    imageio.imwrite(f"{train_path}/r_{i:04d}.png", image)
    

    # openglcsv
    # opengl_transform_file = (f"{source_path}/opengltransform.csv")
    # file_exists = os.path.isfile(opengl_transform_file)
    # opengl_quaternion = rotation_2_quaternion(opengltransform[:3, :3])
    # opengl_translation = opengltransform[:3, -1]

    # with open(opengl_transform_file, mode='a', newline='') as f:
    #     writer = csv.writer(f)
    #     if not file_exists:
    #         writer.writerow(['index','x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    #     opengl_pose = [i ,*opengl_translation, *opengl_quaternion]
    #     writer.writerow(opengl_pose)


if __name__ == "__main__":
    rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser) #args: model_path
    # # parser.add_argument("--starting", action="store_true", help="planner_type")
    # parser.add_argument("--planner_type", "-P", type=str, default="random", help="planner_type")
    parser.add_argument("--radius_start", type=float, default=3.0, help="radius range")
    parser.add_argument("--radius_end", type=float, default=10.0, help="radius range")
    parser.add_argument("--phi_min", type=float, default=0.2617, help="the minimum of phi in uniform_sampling")
    parser.add_argument("--sampling_method", type=str, default="random", help="options: random, circular")
    parser.add_argument("--set_type", type=str, default="train", help="train set or test set")
    parser.add_argument("--experiment_path", type=str, required=True, help="must be defined in evaluation mode")
    parser.add_argument("--views_num", type=int, default="5", help="the number of candidate views")
    parser.add_argument("--sort_view", action="store_true", help="sort_view")
    parser.add_argument("--record", action="store_true", help="record time")
    parser.add_argument("--time_budget", type=float, default=300.0, help="the time that update the path")
    parser.add_argument("--scheduling_window", type=float, default=20.0, help="the time that update the path")
    parser.add_argument("--scheduling_num", type=float, default=3.0, help="the number of the time of scheduling window")
    parser.add_argument("--initial_training_time", type=float, default=1.0, help="the time for first 3DGS trainin")
    parser.add_argument("--experiment_id", type=int, default=1, help="experiment id")
    args = parser.parse_args()

    
    # if 360 % args.views_num !=0:
    #     print(f"view_num: {args.views_num} is invalid, the number must be divisible by 360")
    #     exit()

    print(
        f"---------- random planner ----------\n"
    )
    planner_cfg_path = os.path.join(
        root_dir, "planner/config", f"random_planner.yaml"
    )
    print(planner_cfg_path)
    assert os.path.exists(planner_cfg_path)
    with open(planner_cfg_path, "r") as config_file:
        planner_cfg = yaml.safe_load(config_file)
    # planner_cfg.update(__dict__)
    planner_cfg["planner_type"] = "random"
    planner_cfg["experiment_path"] = args.experiment_path
    planner_cfg["experiment_id"] = args.experiment_id 
    planner_cfg["radius_start"] = args.radius_start
    planner_cfg["radius_end"] = args.radius_end
    planner_cfg["sampling_method"] = args.sampling_method
    planner_cfg["sampling_num"] = args.views_num
    planner_cfg["scheduling_num"] = args.scheduling_num

    sampling_method = args.sampling_method
    sampling_num = args.views_num

    print(planner_cfg)
    nbv_planner = get_planner(planner_cfg)
    camera_info = nbv_planner.simulator_bridge.camera_info

    if args.set_type == "train":
        create_empty_test_set(args.experiment_path, camera_info)

    # if sampling_method == "random":
    #     main_size = int(args.views_num) * int(args.scheduling_num)
    #     main_view_list = np.empty((main_size+100, 3))
    #     for i in range(main_size+100):
    #         main_view_list[i] = uniform_sampling(args.radius_start, args.radius_end, args.phi_min,i)
    # elif sampling_method == "circular":
    if sampling_method == "circular":    
        main_view_list = sphere_sampling_unorder(longtitude_range = 16, latitude_range = 4,
                                    radius_start = args.radius_start, radius_end = args.radius_end) 


    # move to the all candidate views
    timer_start = time.time()
    pose_idxs = 0
    time_budget = args.time_budget
    while time_budget > 0:   
        # index = int(args.views_num) * updated_time

        if sampling_method  == "random":
            view_list = nbv_planner.sampling_view(sampling_method, sampling_num)
            if pose_idxs == 0:
                phi = 15 * (np.pi/180)
                theta = 0.0
                radius = 4
                init_view = [phi, theta, radius]
                view_list = np.insert(view_list, 0, init_view,axis=0)
        elif sampling_method  == "circular":
            # base = nbv_planner.planning_time * 80
            # if nbv_planner.planning_time > 7:
            #     base = ((nbv_planner.planning_time % 8) - 1) * 80 + 40
            # view_list = main_view_list[base:base+sampling_num]
            view_list = main_view_list

        if args.sort_view or args.set_type == "test":
            sorted_indices = np.lexsort((view_list[:,1], view_list[:,0]))
            view_list = view_list[sorted_indices]
        print(view_list)
        for view in view_list:
            timer_end = time.time()
            if sampling_method  == "random" and (timer_end - timer_start > args.scheduling_window):
                nbv_planner.planning_time +=1
                timer_start = timer_end
                break
            if time_budget > 0:
                flying_start_time = time.time()
                nbv_planner.move_sensor(view)
                time.sleep(2.5) 
                flying_end_time = time.time()
                
                if time_budget - (flying_end_time - flying_start_time) < 0 and pose_idxs > 0:
                    time_budget = 0 
                    continue
                captured_start_time = time.time()
                rgb, depth, captured_pose = nbv_planner.simulator_bridge.get_image()
                save_image_into_train_set(args.experiment_path, captured_pose, rgb, camera_info)
                captured_end_time = time.time()

                if args.record == True and pose_idxs == 0:
                    flying_time_csv, captured_time_csv = setup_csv(args.experiment_path)

                if args.record == True: 
                    with open(flying_time_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([pose_idxs, flying_end_time - flying_start_time])

                    with open(captured_time_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([pose_idxs, captured_end_time - captured_start_time])    
                pose_idxs += 1    
                if pose_idxs == 1:
                    continue
                # print(f"{pose_idxs}: t{time_budget}")
                time_budget = time_budget - (captured_end_time - flying_start_time)