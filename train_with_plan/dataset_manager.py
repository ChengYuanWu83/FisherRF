#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import shutil # [cyw]: testing function: for copy image
import json   # [cyw]: testing function: for rewrite transform.json
import imageio.v2 as imageio
import yaml
import numpy as np
from planner import utils

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_image_with_viewport(source_path, captured_pose, image, camera_info):
    # if image == None:
    #     print("fail to load image")
    #     return
    print("------ record simulator data ------\n")
    # [cyw]:save transform_test.json
    camera_dict = utils.get_camera_json(camera_info) # camera_info json
    transform_json_file = (f"{source_path}/transforms_test.json")
    if not os.path.isfile(transform_json_file):
        with open(transform_json_file, "w") as f:
            json.dump(camera_dict, f, indent=4)  

    #[cyw]: save transform_train.json
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
    transformation = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )  # transform gazebo coordinate to opengl format
    transform = utils.quaternion_to_rotation_matrix(captured_pose)
    print(transform)
    opengltransform = transform @ transformation
    train_image_file = (f"./train/r_{i:04d}")
    data_frame = {
        "file_path": train_image_file,
        "transform_matrix": opengltransform.tolist(),
    }
    record_dict["frames"].append(data_frame)
    with open(transform_json_file, "w") as f:
        json.dump(record_dict, f, indent=4)
    #[cyw]: save the image
    train_path = os.path.join(source_path, "train")
    os.makedirs(train_path, exist_ok=True)
    imageio.imwrite(f"{train_path}/r_{i:04d}.png", image)
    
    #[cyw]: save camera_info yaml file
    # with open(f"{source_path}/camera_info.yaml", "w") as f:
    #     yaml.safe_dump(camera_info, f)

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     parser.add_argument('--ip', type=str, default="127.0.0.1")
#     parser.add_argument('--port', type=int, default=6009)
#     parser.add_argument('--debug_from', type=int, default=-1)
#     parser.add_argument('--detect_anomaly', action='store_true', default=False)
#     parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
#     parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000])
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
#     parser.add_argument("--start_checkpoint", type=str, default = None)
#     args = parser.parse_args(sys.argv[1:])
#     args.save_iterations.append(args.iterations)
    
#     print("Optimizing " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     # Start GUI server, configure and run training
#     network_gui.init(args.ip, args.port)
#     torch.autograd.set_detect_anomaly(args.detect_anomaly)

#     save_image(dataset, captured_pose, image, camera_info)

#     # All done
#     print("\nTraining complete.")
