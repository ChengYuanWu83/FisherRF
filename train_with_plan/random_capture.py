
import rospy
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from train_with_plan.dataset_manager import save_image_with_viewport
from argparse import ArgumentParser, Namespace
from planner import get_planner
from planner.utils import uniform_sampling, sphere_sampling
import numpy as np
import yaml
import time
from planner import utils
import json
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
import csv
    
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


def save_image_into_set(source_path, captured_pose, image, camera_info, set_type):
    print("------ record simulator data ------\n")
    # [cyw]:save transform_{set_type}.json
    camera_dict = utils.get_camera_json(camera_info) # camera_info json
    transform_json_file = (f"{source_path}/transforms_{set_type}.json")
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

    set_image_file = (f"./{set_type}/r_{i:04d}")
    data_frame = {
        "file_path": set_image_file,
        "transform_matrix": opengltransform.tolist(),
    }
    record_dict["frames"].append(data_frame)
    with open(transform_json_file, "w") as f:
        json.dump(record_dict, f, indent=4)
    #[cyw]: save the image
    set_path = os.path.join(source_path, set_type)
    os.makedirs(set_path, exist_ok=True)
    imageio.imwrite(f"{set_path}/r_{i:04d}.png", image)


if __name__ == "__main__":
    rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser) #args: model_path
    # # parser.add_argument("--starting", action="store_true", help="planner_type")
    # parser.add_argument("--planner_type", "-P", type=str, default="random", help="planner_type")
    parser.add_argument("--radius", type=float, default=2.0, help="radius of uniform_sampling")
    parser.add_argument("--phi_min", type=float, default=0.26, help="the minimum of phi in uniform_sampling")
    parser.add_argument("--experiment_path", type=str, default="not defined", help="must be defined in evaluation mode")
    parser.add_argument("--candidate_views_num", type=int, default="50", help="the number of candidate views")
    parser.add_argument("--set_type", type=str, default="train", help="train set or test set")
    parser.add_argument("--record", action="store_true", help="record time")
    parser.add_argument("--time_budget", type=float, default=100.0, help="time budget")
    parser.add_argument("--sort_view", action="store_true", help="sort_view")
    parser.add_argument("--sampling_method", type=str, default="random", help="options: random, sphere")
    parser.add_argument("--radius_start", type=float, default=2.0, help="radius range")
    parser.add_argument("--radius_end", type=float, default=5.0, help="radius range")
    args = parser.parse_args()

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
    planner_cfg["action_space"]["radius"] = args.radius
    # planner_cfg["experiment_id"] = args.experiment_id #[cyw]: 
    print(planner_cfg)
    nbv_planner = get_planner(planner_cfg)
    camera_info = nbv_planner.simulator_bridge.camera_info

    view_list = None
    if args.sampling_method == "random":
        view_list = np.empty((args.candidate_views_num, 3))
        for i in range(args.candidate_views_num):
            view_list[i] = uniform_sampling(args.radius_start, args.radius_end, args.phi_min)
    elif args.sampling_method == "sphere":
        view_list = sphere_sampling(longtitude_range = 16, latitude_range = 4,
                                    radius_start = args.radius_start, radius_end = args.radius_end) 
        np.random.shuffle(view_list)
    else:
        print(f"unknown sampling method {args.sampling_method}")

    # sort the view to enhance the flying time
    if args.sort_view or args.set_type == "test":
        sorted_indices = np.argsort(view_list[:, 1])
        view_list = view_list[sorted_indices]

    print(view_list)

    pose_idxs = 0
    time_budget = args.time_budget
    # move to the all candidate views
    for view in view_list:
        if time_budget > 0 or args.set_type == "test":
            flying_start_time = time.time()
            nbv_planner.move_sensor(view)
            flying_end_time = time.time()
            # time.sleep(2.5)
            captured_start_time = time.time()
            rgb, depth, captured_pose = nbv_planner.simulator_bridge.get_image()
            save_image_into_set(args.experiment_path, captured_pose, rgb, camera_info, args.set_type)
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
            
            time_budget = time_budget - (captured_end_time - flying_start_time)