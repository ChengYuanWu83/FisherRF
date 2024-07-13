
import rospy
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from train_with_plan.dataset_manager import save_image_with_viewport
import yaml
from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams, OptimizationParams
from planner import get_planner
from planner.utils import uniform_sampling, xyz_to_view, random_view
import numpy as np
import time
import json

def plan(planner_type, experiment_path, candidate_views_num, radius):

    # experiment_path = args.experiment_path
    # planner_type = args.planner_type
    #[cyw]: initialize planner
    # find planner configuration file
    print(
        f"---------- {planner_type} planner ----------\n"
    )
    planner_cfg_path = os.path.join(
        root_dir, "planner/config", f"{planner_type}_planner.yaml"
    )
    print(planner_cfg_path)
    assert os.path.exists(planner_cfg_path)
    with open(planner_cfg_path, "r") as config_file:
        planner_cfg = yaml.safe_load(config_file)
    # planner_cfg.update(__dict__)
    planner_cfg["planner_type"] = planner_type
    planner_cfg["experiment_path"] = experiment_path
    # planner_cfg["experiment_id"] = args.experiment_id #[cyw]: 
    print(planner_cfg)
    nbv_planner = get_planner(planner_cfg)
    # candidate_views_num = nbv_planner.num_candidates
    # setting the starting view

    # check if the transform exist
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

    #[cyw]: candidates view generator
    for i in range(candidate_views_num):
        view_list[i] = random_view(
            current_position, #[cyw]: change planner pose to ros pose, oringinal: self.current_pose[:3, 3]
            nbv_planner.radius,
            nbv_planner.phi_min,
            min_view_change=0.2,
            max_view_change=nbv_planner.view_change,
        )
    sorted_indices = np.argsort(view_list[:, 1])
    view_list = view_list[sorted_indices]

    if start == True and i == 0:
        view_list[0] = starting_view
    # move to the all candidate views
    for view in view_list:
        nbv_planner.move_sensor(view)
        time.sleep(2.5)
        rgb, depth, captured_pose = nbv_planner.simulator_bridge.get_image()
        camera_info = nbv_planner.simulator_bridge.camera_info
        save_image_with_viewport(experiment_path, captured_pose, rgb, camera_info)



if __name__ == "__main__":
    rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument("--starting", action="store_true", help="planner_type")
    parser.add_argument("--radius", type=float, default=2.0, help="radius of uniform_sampling")
    parser.add_argument("--planner_type", "-P", type=str, default="random", help="planner_type")
    parser.add_argument("--experiment_path", type=str, default="not defined", help="must be defined in evaluation mode")
    parser.add_argument("--candidate_views_num", type=int, default="10", help="the number of candidate views")

    args = parser.parse_args()
    plan(args.planner_type, args.experiment_path, args.candidate_views_num, args.radius)
