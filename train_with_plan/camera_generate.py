import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from argparse import ArgumentParser
import numpy as np
from planner.utils import quaternion_to_rotation_matrix, xyz_to_view, random_view, view_to_position_and_rotation, focal_len_to_fov, view_to_pose
import json
import csv

def plan(experiment_path, candidate_views_num, radius):
    # [cyw]: camera parameter is get from simulator, so i pre-defined it here
    resolution = [800.0, 800.0]
    c = [400.0, 400.0]
    focal = [692.8203125, 692.8203125]

    fov = focal_len_to_fov(focal, resolution)

    camera_dict = {}
    camera_dict["camera_angle_x"] = fov[0]
    camera_dict["camera_angle_y"] = fov[1]
    camera_dict["fl_x"] = focal[0]
    camera_dict["fl_y"] = focal[1]
    camera_dict["k1"] = 0.000001
    camera_dict["k2"] = 0.000001
    camera_dict["p1"] = 0.000001
    camera_dict["p2"] = 0.000001
    camera_dict["cx"] = c[0]
    camera_dict["cy"] = c[1]
    camera_dict["w"] = resolution[1]
    camera_dict["h"] = resolution[0]
    camera_dict["frames"] = []
    camera_dict["scale"] = 1.0
    camera_dict["aabb_scale"] = 2.0


    os.makedirs(experiment_path, exist_ok=True)

    train_path = os.path.join(experiment_path, "train")
    os.makedirs(train_path, exist_ok=True)

    transform_file = (f"{experiment_path}/transforms_train.json")
    start = not os.path.isfile(transform_file)
    if start:
        with open(transform_file, "w") as f:
            json.dump(camera_dict, f, indent=4)
            record_dict = camera_dict
        candidate_views_num +=1
        starting_view_setting = np.array([1.0, 0.0, 0.5])
        starting_view = xyz_to_view(xyz=starting_view_setting, radius=radius)
        current_position = starting_view_setting
    else:
        with open(transform_file, 'r') as f:
            record_dict = json.load(f)
        transform_matrix = record_dict["frames"][-1]["transform_matrix"]
        current_position = np.array(transform_matrix)[:3, -1]
    view_list = np.empty((candidate_views_num, 2))
    
    min_height = 0.3
    phi_min = np.arcsin(min_height /radius)
    for i in range(candidate_views_num):
        view_list[i] = random_view(
            current_position, #[cyw]: change planner pose to ros pose, oringinal: self.current_pose[:3, 3]
            radius,
            phi_min,
            min_view_change=0.2,
            max_view_change=1.05, # 60 degree = 1.05 radian
        )
    sorted_indices = np.argsort(view_list[:, 1])
    view_list = view_list[sorted_indices]

    if start == True and i == 0:
        view_list[0] = starting_view

    csv_file = (f"{experiment_path}/view_list.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['index','x', 'y', 'z', 'qx', 'qy', 'qz'])

        for i, view in enumerate(view_list):
            translation, rotation = view_to_position_and_rotation(view, radius) #[cyw]: pose format
            #write json
            pose_matrix = view_to_pose(view, radius)
            j = len(record_dict["frames"])
            transformation = np.array(
            [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )  # transform gazebo coordinate to opengl format
            opengltransform = pose_matrix @ transformation
            train_image_file = (f"./train/r_{j:04d}")
            data_frame = {
                "file_path": train_image_file,
                "transform_matrix": opengltransform.tolist(),
            }
            record_dict["frames"].append(data_frame)
            with open(transform_file, "w") as f:
                json.dump(record_dict, f, indent=4)
            #write csv
            pose = [i ,*translation, *rotation]
            writer.writerow(pose)
            print(pose)

if __name__ == "__main__":
    # rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument("--starting", action="store_true", help="planner_type")
    parser.add_argument("--radius", type=float, default=3.0, help="radius of uniform_sampling")
    # parser.add_argument("--planner_type", "-P", type=str, default="random", help="planner_type")
    parser.add_argument("--experiment_path", type=str, default="not defined", help="must be defined in evaluation mode")
    parser.add_argument("--candidate_views_num", type=int, default="10", help="the number of candidate views")


    args = parser.parse_args()
    plan(args.experiment_path, args.candidate_views_num, args.radius)
