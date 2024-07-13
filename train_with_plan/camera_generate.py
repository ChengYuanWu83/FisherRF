import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from argparse import ArgumentParser
import numpy as np
from planner.utils import uniform_sampling, xyz_to_view, random_view, view_to_pose
import json

def plan(experiment_path, candidate_views_num, radius):
    # [cyw]: generate the view 
    transform_file = (f"{experiment_path}/transforms_train.json")
    start = not os.path.isfile(transform_file)
    if start:
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

    for view in view_list:
        pose = view_to_pose(view, radius) #[cyw]: pose format
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